# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
DIR="/mnt/ILSVRC2017_VID/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000"
ARCH="alexnet"
LR=0.05
WD=-5
K=30
WORKERS=12
EXP="/home/${USER}/test/exp"
PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"
EPOCHS=100

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --epoch ${EPOCHS}$
#/miniconda/envs/vmr/bin/python


