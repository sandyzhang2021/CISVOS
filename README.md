#**Semi-supervised Video Object Segmentation with Causal Mechanism**
This repository is the official implementation of semi-supervised video object segmentation with causal mechanism.
## 1. Requirements
We built and tested the repository on Python 3.8 and pytorch1.12.1 with two NVIDIA V100 (32GB) 
To install requirements, run:
```bash
pip3 install -r requirements.txt
```
## 2. Training Causal CAM and IRNet
python3 run_sample.py --root your dataset root

## 3.Training SVOS
python3 train.py --level 1 --new --resume pretrained/*.pth --dataset dataset val_root --lr 4e-6 --scheduler-step 200 --total-epoch 1000 --log

## 4.Eval 
python3 train.py ---level 1 --resume pretrained/*.pth --dataset dataset test_root --viz 

## 5.Evaluation 
python3 evaluation_method.py --task semi-supervised --results_path dataset val_root
