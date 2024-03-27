import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch

from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import dataloader
import numpy as np
import myutils
import imageio
import importlib
from dataloader import CAT_NAME_TO_NUM
from misc import torchutils, imutils

cudnn.enabled = True



def _work(process_id, model, dataset, args):
    
    databin = dataset[process_id]
    
    
    n_gpus = torch.cuda.device_count()
    
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        
        for iter, pack in enumerate(data_loader):
            # name_list,images_names = dataloader.img_name_list(root=args.root,name=pack['name'][0])
            img_name = pack['name'][0]
            imgs = pack['img']
            label = pack['label']
            size = pack['size']
            # size=(960,960)
            stride_CAM = []
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in imgs]
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)
            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
            
            img_name_dir = os.path.join(args.cam_out_dir,img_name)
            print("img_name_dir:",img_name_dir)
            if not os.path.exists(img_name_dir):
                os.makedirs(img_name_dir)
            np.save(os.path.join(img_name_dir,img_name+'.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu()})
            # np.save(os.path.join(img_name_dir,img_name+'.npy'),
            #         {"keys": valid_cat, "cam": stride_CAM,"high_res":highres_cam})
            # cam_dict =np.load(os.path.join(img_name_dir,img_name+'.npy'), allow_pickle=True)
            # cams = cam_dict[()]['cam']
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

    print("It ok!")


def run(args):
    torch.cuda.empty_cache()
    
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name ), strict=True)
    # model = torch.nn.DataParallel(model).cuda() 
    # model = model.cuda()#
    model.eval()

    n_gpus = torch.cuda.device_count()
    # n_gpus = 4

    # dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
    #                                                          voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = dataloader.VOC12ClassificationDatasetMSF(root=args.root, img_name_list_path=args.train_list,scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)
        
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()