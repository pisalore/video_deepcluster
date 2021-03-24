# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
DATA_TYPE="train"
DATA_DIR="/thecube/students/${USER}/ILSVRC2017_VID/ILSVRC/Data/VID/${DATA_TYPE}/"
ANN_DIR="/thecube/students/${USER}/ILSVRC2017_VID/ILSVRC/Annotations/VID/${DATA_TYPE}/"
LBL_TXT="/home/${USER}/video_deepcluster/map_vid.txt"
EXP="/thecube/students/${USER}/"

PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"

${PYTHON} clean_data.py ${DATA_DIR} ${ANN_DIR} ${LBL_TXT} --output ${EXP} --data_type ${DATA_TYPE}
#/miniconda/envs/vmr/bin/python

