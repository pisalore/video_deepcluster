# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
PKL="/thecube/students/${USER}/vid_dataset_2021.3.10_18_0_46.pkl"
LABELS="/thecube/students/${USER}/labels.pkl"
STEP=10
ARCH="alexnet"
LR=0.05
WD=-5
WORKERS=12
EXP="/thecube/students/${USER}/test_step_${STEP}_K_${K}/exp"
PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"
EPOCHS=100
BATCH=256
mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 ${PYTHON}  vid_classifier.py --load_step ${STEP} --batch ${BATCH} --arch ${ARCH} \
--lr ${LR} --wd ${WD} --dataset_pkl ${PKL} --labels_pkl ${LABELS} --sobel --verbose --workers ${WORKERS} --epoch ${EPOCHS}
#/miniconda/envs/vmr/bin/python