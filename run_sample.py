import argparse
import os
# import dataloader
import numpy as np
import glob
from misc import pyutils
from collections import defaultdict
import myutils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    # parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    # parser.add_argument("--root", required=True, type=str,default='./dataset/DAVIS17/trainval',
    #                     help="Path to DAVIS2017, must contain ./dataset/DAVIS17/trainval.")
    parser.add_argument("--root", required=True, type=str,default='./dataset/YouTube/train',
                        help="Path to DAVIS2017, must contain ./dataset/DAVIS17/trainval.")

    # Dataset
    parser.add_argument("--train_list", default="2017/train.txt", type=str)
    parser.add_argument("--val_list", default="2017/val.txt", type=str)
    parser.add_argument("--img_set", default="2017/train.txt", type=str)
    parser.add_argument("--infer_list", default="2017/val.txt", type=str,
                        help="2017/train.txt to train a fully supervised model, "
                             "2017/train.txt or 2017/test-dev.txt to quickly check the quality of the labels.")
    # parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="model.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=400, type=int)
    parser.add_argument("--cam_batch_size", default=1, type=int)
    parser.add_argument("--cam_num_epoches", default=500, type=int)
    parser.add_argument("--cam_learning_rate", default=0.01, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    # parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="model.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=400, type=int)
    parser.add_argument("--irn_batch_size", default=1, type=int)
    parser.add_argument("--irn_num_epoches", default=100, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.1)
    # parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--out", default="cls_labels2.npy", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    # parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--make_cam_pass", default=True)
    parser.add_argument("--train_irn_pass", default=True)
    parser.add_argument("--make_ins_seg_pass", default=True)
   

    args = parser.parse_args()
    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)


    pyutils.Logger(args.log_name + '.log')
    print(vars(args))
    if args.train_cam_pass is True:
        import train_cam
        timer = pyutils.Timer('train_cam:')
        train_cam.run(args)
        # import train_cam_yv
        # timer = pyutils.Timer('train_cam:')
        # train_cam.run(args)
        

    if args.make_cam_pass is True:
        import make_cam

        timer = pyutils.Timer('make_cam:')
        make_cam.run(args)

    # if args.eval_cam_pass is True:
    #     import eval_cam

    #     timer = pyutils.Timer('eval_cam:')
    #     eval_cam.run(args)

    # if args.cam_to_ir_label_pass is True:
    #     import cam_to_ir_label

    #     timer = pyutils.Timer('cam_to_ir_label:')
    #     cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import train_irn

        timer = pyutils.Timer('train_irn:')
        train_irn.run(args)

    if args.make_ins_seg_pass is True:
        import make_ins_seg_labels

        timer = pyutils.Timer('make_ins_seg_labels:')
        make_ins_seg_labels.run(args)

    # if args.eval_ins_seg_pass is True:
    #     import eval_ins_seg

    #     timer = pyutils.Timer('eval_ins_seg:')
    #     eval_ins_seg.run(args)

    # if args.make_sem_seg_pass is True:
    #     import make_sem_seg_labels

    #     timer = pyutils.Timer('step.make_sem_seg_labels:')
    #     make_sem_seg_labels.run(args)

    # if args.eval_sem_seg_pass is True:
    #     import eval_sem_seg

    #     timer = pyutils.Timer('step.eval_sem_seg:')
    #     eval_sem_seg.run(args)