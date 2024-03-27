import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import dataloader
from misc import pyutils, torchutils, indexing
import importlib
import argparse

def run(args):
   
    path_index = indexing.PathIndex(radius=10, default_size=(args.irn_crop_size // 4, args.irn_crop_size // 4))

    model = getattr(importlib.import_module(args.irn_network), 'AffinityDisplacementLoss')(
        path_index)

    train_dataset = dataloader.VOC12AffinityDataset(root=args.root,img_name_list_path=args.train_list,
                                                          label_dir=args.ir_label_out_dir,
                                                          indices_from=path_index.src_indices,
                                                          indices_to=path_index.dst_indices,
                                                          hor_flip=True,
                                                          crop_size=args.irn_crop_size,
                                                          crop_method="random",
                                                          rescale=(0.5, 1.5)
                                                          )
    train_data_loader = DataLoader(train_dataset, batch_size=args.irn_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // args.irn_batch_size) * args.irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.irn_learning_rate, 'weight_decay': args.irn_weight_decay}
    ], lr=args.irn_learning_rate, weight_decay=args.irn_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.irn_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):
            imgs = pack['img']
            pos_aff_loss = []
            neg_aff_loss = []
            dp_fg_loss = []
            dp_bg_loss= []
            bg_pos_label = pack['aff_bg_pos_label']
            fg_pos_label = pack['aff_fg_pos_label']
            neg_label = pack['aff_neg_label']
            i = 0
            total_loss = []
            Total_loss = 0
            for img in imgs:
                img = img.cuda()
                pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)
                fg_pos_label[i] = fg_pos_label[i].cuda()
                fg_pos_aff_loss = torch.sum(fg_pos_label[i] * pos_aff_loss) / (torch.sum(fg_pos_label[i]) + 1e-5)
                bg_pos_label[i] = bg_pos_label[i].cuda()
                bg_pos_aff_loss = torch.sum(bg_pos_label[i] * pos_aff_loss) / (torch.sum(bg_pos_label[i]) + 1e-5)
                pos_aff_loss = pos_aff_loss.cuda()
                pos_aff_loss = bg_pos_aff_loss/ 2 + fg_pos_aff_loss / 2#
                pos_aff_loss = fg_pos_aff_loss + fg_pos_aff_loss

                neg_label[i] = neg_label[i].cuda()
                neg_aff_loss = torch.sum(neg_label[i] * neg_aff_loss) / (torch.sum(neg_label[i]) + 1e-5)

                dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label[i], 1)) / (2 * torch.sum(fg_pos_label[i]) + 1e-5)

                dp_bg_loss = torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label[i], 1)) / (2 * torch.sum(bg_pos_label[i]) + 1e-5)
                avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})
                total = (pos_aff_loss + neg_aff_loss) / 2 + (dp_fg_loss + dp_bg_loss) / 2
                total_loss.append(total)
                i= i+ 1
            Total_loss = sum(total_loss)
            optimizer.zero_grad()
            Total_loss.backward()
            optimizer.step()
            

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (Total_loss),
                      'imps:%.1f' % ((iter + 1) * args.irn_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
            else:
                timer.reset_stage()

    # infer_dataset = dataloader.VOC12ImageDataset(args.infer_list,
    #                                                    root=args.root,
    #                                                    crop_size=args.irn_crop_size,
    #                                                    crop_method="top_left")
    infer_dataset = dataloader.VOC12ImageDataset2(root=args.root,
                                                 img_name_list_path=args.infer_list,
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.irn_batch_size,
                                   shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            imgs = pack['img']
            for img in imgs:
                img = img.cuda()
                aff, dp = model(img, False)
            # aff, dp = model(img, False)

                dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())
        model.module.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
            # model.mean_shift.running_mean.append(a)
    print('done.')

    torch.save(model.module.state_dict(), args.irn_weights_name)
    torch.cuda.empty_cache()
