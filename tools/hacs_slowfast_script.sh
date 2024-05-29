#!/bin/bash
echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/hacs_slowfast.yaml --output pretrained
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/hacs_slowfast.yaml ckpt/hacs_slowfast_pretrained/epoch_040.pth.tar

# CUDA_VISIBLE_DEVICES=1 python train.py ./configs/hacs_slowfast.yaml --output allepoch11_warm4_backboneconv

# CUDA_VISIBLE_DEVICES=1 python eval.py ./configs/hacs_slowfast.yaml ckpt/hacs_slowfast_allepoch40_warm5/epoch_039.pth.tar