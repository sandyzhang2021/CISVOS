import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader,Dataset
# from torch.utils.data import Dataset
import torch.nn.functional as F
import argparse
import importlib
import os
import cv2
import numpy as np
import dataloader
from misc import pyutils, torchutils
from misc.pyutils import class_balanced_cross_entropy_loss
import myutils
from model.resnet50 import resnet50
# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM,EigenGradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
# from pytorch_grad_cam import GuidedBackpropReLUModel

def validate(model, data_loader):
        print('validating ... ', flush=True, end='')

        val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

        model.eval()
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        LOSS2 = 0
        with torch.no_grad():
            for pack in data_loader:
                name = pack['name']
                print("name:",name)
                imgs = pack['img']
                label = pack['label'].cuda(non_blocking=True)
                # print("img.shape:",img.shape)
                # print("label2:",label)# torch.Size([1, 60])
                
                # x=[model(img.cuda(non_blocking=True))# torch.Size([2, 60])
                            # for img in imgs]
                # loss1 =  torch.nn.BCEWithLogitsLoss()(x[0], label)
                # loss1 = F.multilabel_soft_margin_loss(torch.Tensor(x[0]), label)
                # Loss1 = []
                loss = 0
                for img in imgs:
                    x = model(img.cuda(non_blocking=True))
                    loss2 = F.multilabel_soft_margin_loss(torch.Tensor(x[0]), label)
                    # loss2 = torch.nn.BCEWithLogitsLoss()(x, label)
                    # loss2 = torch.nn.CrossEntropyLoss()(x, label)
                    loss = loss + loss2
            LOSS2 = LOSS2 + loss
            LOSS2 = torch.Tensor(LOSS2)
            print("loss2:",LOSS2.item()) 
            val_loss_meter.add({'loss2': LOSS2.item()})

        model.train()

        print('loss: %.4f' % (val_loss_meter.pop('loss2')))

        return


def run(args):
        # import os
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
        # torch.cuda.empty_cache()
        model = getattr(importlib.import_module(args.cam_network), 'Net')()

        # criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion = torch.nn.BCELoss().cuda()
        train_dataset = dataloader.VOC12ClassificationDataset(root=args.root,img_name_list_path=args.train_list,
                                                                resize_long=(400, 400), hor_flip=True,
                                                                crop_size=400, crop_method="random")
        
        train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

        val_dataset = dataloader.VOC12ClassificationDataset(root=args.root,img_name_list_path=args.val_list,
                                                              crop_size=400)
        val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        param_groups = model.trainable_parameters()
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
            {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

        # model = torch.nn.DataParallel(model).cuda()#
        model = model.cuda()#
        model.train()


        avg_meter = pyutils.AverageMeter()

        timer = pyutils.Timer()
        
        for ep in range(args.cam_num_epoches):

            print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
            LOSS = 0
            for step,pack in enumerate(train_data_loader):
                name = pack['name']
                print("name:",name)
                imgs = pack['img']
                label = pack['label'].cuda(non_blocking=True)
                loss = 0
                for img in imgs:
                    x = model(img.cuda(non_blocking=True))
                    # x = torch.argmax(x, dim=0).long()
                    loss1 = F.multilabel_soft_margin_loss(x[0], label)
                    
                    # loss1 = torch.nn.BCEWithLogitsLoss()(x, label)
                    # loss1 = criterion(x[0], label)#BCELoss
                    # loss1 = torch.nn.CrossEntropyLoss()(x, label)
                    loss = loss + loss1
            LOSS = LOSS + loss
                # loss = torch.Tensor(loss)
            LOSS = torch.Tensor(LOSS)
            print("loss1:",LOSS.item())         
            avg_meter.add({'loss1': LOSS.item()})

            optimizer.zero_grad()
                # loss.requires_grad_(True)  # 
                # loss.backward(retain_graph=True)  
            LOSS.backward()
            optimizer.step()

            if (optimizer.global_step-1)%100 == 0:
                    timer.update_progress(optimizer.global_step / max_step)

                    print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

            else:
                    validate(model, val_data_loader)
                    timer.reset_stage()

        torch.save(model.state_dict(), args.cam_weights_name )
        print(myutils.gct(), 'Train_cam done.')
        torch.cuda.empty_cache()
