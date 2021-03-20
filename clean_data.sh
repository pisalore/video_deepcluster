# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
DATA_TYPE="val"
DATA_DIR="/mnt/ILSVRC2017_VID/ILSVRC/Data/VID/${DATA_TYPE}/"
ANN_DIR="/mnt/ILSVRC2017_VID/ILSVRC/Annotations/VID/${DATA_TYPE}/"
EXP="/thecube/students/${USER}/"

PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"

${PYTHON} data-cleaner.py ${DATA_DIR} ${ANN_DIR} --output ${EXP} --data_type ${DATA_TYPE}
#/miniconda/envs/vmr/bin/python

