# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
#TODO: paths reworking
PKL_TRAIN="/thecube/students/${USER}/vid_dataset_train_2021.3.25_16_47_0.pkl"
PKL_VAL="/thecube/students/${USER}/vid_dataset_val_2021.3.25_4_46_56.pkl"

NUM_OUT_CLASSES=30

STEP=3
K=300
ARCH="alexnet"
LR=0.05
WD=-5
WORKERS=12
EXP="/thecube/students/${USER}/test_step_${STEP}_K_${K}/exp/fine_tuning"
PRE_TRAINED_MODEL="/thecube/students/${USER}/test_step_${STEP}_K_${K}/exp/model_s3k300.pth.tar"
PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"
EPOCHS=100
BATCH=256
mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON}  vid_classifier.py --load_step ${STEP} --batch ${BATCH} --arch ${ARCH} \
--lr ${LR} --wd ${WD} --train_dataset_pkl ${PKL_TRAIN} --val_dataset_pkl ${PKL_VAL} --model ${PRE_TRAINED_MODEL} --k ${K} --out_classes ${NUM_OUT_CLASSES} \
--sobel --verbose --workers ${WORKERS} --epoch ${EPOCHS} --exp ${EXP}
#/miniconda/envs/vmr/bin/python