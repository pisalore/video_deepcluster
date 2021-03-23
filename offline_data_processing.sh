# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
PKL="/thecube/students/${USER}/vid_dataset_2021.3.10_18_0_46.pkl"
LABELS="/thecube/students/${USER}/labels.pkl"
IN_DIR="/mnt/ILSVRC2017_VID/"
OUT_DIR="/thecube/students/${USER}/ILSVRC2017_VID2/"
DATA_DIR="/thecube/students/${USER}/ILSVRC2017_VID2/Data/VID/train/"
STEP=3
PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"


CUDA_VISIBLE_DEVICES=0 ${PYTHON} offline_data_processing.py ${PKL} ${LABELS} ${IN_DIR} ${OUT_DIR} ${DATA_DIR} --load_step ${STEP}
#/miniconda/envs/vmr/bin/python