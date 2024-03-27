import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
import cv2
from torch.nn import functional as F
from torch.utils import data
from tensorboardX import SummaryWriter

from dataset import PreTrain_DS, DAVIS_Train_DS, YouTube_Train_DS
from model import AFB_URR, FeatureBank
import myutils
import dataloader

def get_args():
    parser = argparse.ArgumentParser(description='Train AFB')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU card id.')
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='Dataset folder.')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed.')
    parser.add_argument('--log', action='store_true',
                        help='Save the training results.')
    parser.add_argument('--level', type=int, default=0,
                        help='0: pretrain. 1: DAVIS. 2: Youtube-VOS.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate, default 1e-5.')
    parser.add_argument('--lu', type=float, default=0.5,
                        help='Regularization factor, default 0.5.')
    parser.add_argument('--resume', type=str,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--new', action='store_true',
                        help='Train the model from the begining.')
    parser.add_argument('--scheduler-step', type=int, default=25,
                        help='Scheduler step size. Default 25.')
    parser.add_argument('--total-epochs', type=int, default=100,
                        help='Total running epochs. Default 100.')
    parser.add_argument('--budget', type=int, default=300000,
                        help='Max number of features that feature bank can store. Default: 300000')
    parser.add_argument('--obj_n', type=int, default=3,
                        help='Max number of objects that will be trained at the same time.')
    parser.add_argument('--clip_n', type=int, default=10,
                        help='Max frames that will be sampled as a batch.')

    return parser.parse_args()

def train_model(model, dataloader1, criterion, optimizer, desc):
    # global_cams = torch.from_numpy(np.load(os.path.join(
    #     './pretrained', 'global_cam_by_class.npy'))).cuda(non_blocking=True)
    # global_cams = F.interpolate(torch.unsqueeze(global_cams, 0), 4, mode='bilinear', align_corners=False)
    # global_cams = global_cams.detach().cpu().numpy()
    # print("global_cams",global_cams.shape)
    # global_cams = np.mean(global_cams.reshape(-1,4,128,128),axis=1)
    # global_cams = np.expand_dims(global_cams, axis=1)
    # global_cams = np.concatenate((global_cams, global_cams, global_cams), axis=1)#由单通道扩展成三通道
    
    
    # print("global_cams.shape=",global_cams.shape)
    
    # channel_one = global_cams[:,0,:,:]
    # channel_two = global_cams[:,1,:,:]
    # channel_three = global_cams[:,2,:,:]
 
    # channel_one = np.pad(channel_one, ((136, 136),(136, 136),(136, 136)), 'constant')
    # channel_two = np.pad(channel_two, ((136, 136),(136, 136),(136, 136)), 'constant')
    # channel_three = np.pad(channel_three, ((136, 136),(136, 136),(136, 136)), 'constant')
    # # global_cams = np.vstack((channel_one,channel_two,channel_three))
    # global_cams= [channel_one,channel_two,channel_three]
   
    # global_cams=torch.Tensor(global_cams).to(device)
    # print("global_cams.shape=",global_cams.shape)
    
    stats = myutils.AvgMeter()
    uncertainty_stats = myutils.AvgMeter()

    progress_bar = tqdm(dataloader1, desc=desc)
    for iter_idx, sample in enumerate(progress_bar):
        frames, masks, obj_n, info,npys = sample
        obj_n = obj_n.item()
        if obj_n == 1:
            continue

        frames, masks ,npys = frames[0].to(device), masks[0].to(device) ,npys.to(device)
        # frames, masks = frames[0].to(device), masks[0].to(device)

        fb_global = FeatureBank(obj_n, args.budget, device)
        k4_list, v4_list = model.memorize(frames[0:1], masks[0:1])#
        fb_global.init_bank(k4_list, v4_list)
        
        # print("scores.shape=",scores.shape)
        # print("npys.shape=",npys.shape)
        #Causal Intervention 
    
        scores, uncertainty = model.segment(frames[1:], fb_global)
        
        # scores = scores.unsqueeze(2).unsqueeze(2) * npys

        # scores = myutils.mean_agg(scores, r=1)
        
        # Abalation study
        
        # scores = scores.unsqueeze(2).unsqueeze(2) + scores.unsqueeze(2).unsqueeze(2) * npys #
        # scores =  args.lu *scores.unsqueeze(2).unsqueeze(2) * npys  #
        scores = scores.unsqueeze(2).unsqueeze(2) + npys   #
        scores = myutils.mean_agg(scores, r=1)
        #
        label = torch.argmax(masks[1:], dim=1).long()
        optimizer.zero_grad()
        loss = criterion(scores, label)
        loss = loss + args.lu * uncertainty
        loss.backward()
        optimizer.step()

        uncertainty_stats.update(uncertainty.item())
        stats.update(loss.item())
        progress_bar.set_postfix(loss=f'{loss.item():.5f} ({stats.avg:.5f} {uncertainty_stats.avg:.5f})')

        # For debug
        print(info)
        # myutils.vis_result(frames, masks, scores)

    progress_bar.close()

    return stats.avg


def main():
    
    if args.level == 0:
        dataset = PreTrain_DS(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
        desc = 'Pre Train'
    elif args.level == 1:
        dataset = DAVIS_Train_DS(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
        desc = 'Train DAVIS17'
    elif args.level == 2:
        dataset = YouTube_Train_DS(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
        desc = 'Train YV18'
    else:
        raise ValueError(f'{args.level} is unknown.')

    dataloader1 = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    print(myutils.gct(), f'Load level {args.level} dataset: {len(dataset)} training cases.')

    model = AFB_URR(device, update_bank=False, load_imagenet_params=True)
    
 
    model = model.to(device)
    model.train()
    model.apply(myutils.set_bn_eval)  # turn-off BN

    params = model.parameters()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, params), args.lr)

    start_epoch = 0
    best_loss = 100000000
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'], strict=False)
            seed = checkpoint['seed']

            if not args.new:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_loss = checkpoint['loss']
                print(myutils.gct(),
                      f'Loaded checkpoint {args.resume} (epoch: {start_epoch-1}, best loss: {best_loss})')
            else:
                if args.seed < 0:
                    seed = int(time.time())
                else:
                    seed = args.seed
                print(myutils.gct(), f'Loaded checkpoint {args.resume}. Train from the beginning.')
        else:
            print(myutils.gct(), f'No checkpoint found at {args.resume}')
            raise IOError
    else:

        if args.seed < 0:
            seed = int(time.time())
        else:
            seed = args.seed

    print(myutils.gct(), 'Random seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion = torch.nn.CrossEntropyLoss().to(device)#
    # criterion = torch.nn.BCEWithLogitsLoss() #
    # criterion = torch.nn.BCELoss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.5, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.total_epochs):

        lr = scheduler.get_last_lr()[0]
        print('')
        print(myutils.gct(), f'Epoch: {epoch} lr: {lr}')

        loss = train_model(model, dataloader1, criterion, optimizer, desc)
        if args.log:

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'seed': seed,
            }

            checkpoint_path = f'{model_path}/final.pth'
            torch.save(checkpoint, checkpoint_path)

            if best_loss > loss:
                best_loss = loss

                checkpoint_path = f'{model_path}/epoch_{epoch:03d}_loss_{loss:.03f}.pth'
                torch.save(checkpoint, checkpoint_path)

                checkpoint_path = f'{model_path}/best.pth'
                torch.save(checkpoint, checkpoint_path)

                print('Best model updated.')

        scheduler.step()


if __name__ == '__main__':
    # import os
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    print(myutils.gct(), f'Args = {args}')
    torch.multiprocessing.set_start_method('spawn')
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', args.gpu)
    else:
        raise ValueError('CUDA is required. --gpu must be >= 0.')

    if args.log:
        if not os.path.exists('logs'):
            os.makedirs('logs')

        prefix = f'level{args.level}'
        log_dir = 'logs/{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'))
        log_path = os.path.join(log_dir, 'log')
        model_path = os.path.join(log_dir, 'model')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
        # myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
        # myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

        vis_writer = SummaryWriter(log_path)
        vis_writer_step = 0

        print(myutils.gct(), f'Create log dir: {log_dir}')

    main()

    if args.log:
        vis_writer.close()

    print(myutils.gct(), 'Training done.')
