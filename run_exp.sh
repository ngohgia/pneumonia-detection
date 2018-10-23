#!/bin/sh

# CUDA_VISIBLE_DEVCIES=0,1,2,3,4,5,6,7; python train_test.py
CUDA_VISIBLE_DEVCIES=0,1,2,3,4,5,6,7; python train_test_2xdownsample.py
# CUDA_VISIBLE_DEVCIES=0,1,2,3,4,5,6,6; python FocalLOSS_train_test.py
