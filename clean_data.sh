# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
DATA_DIR="/mnt/ILSVRC2017_VID/ILSVRC/Data/VID/train/"
ANN_DIR="/mnt/ILSVRC2017_VID/ILSVRC/Annotations/VID/train/"
PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"

${PYTHON} data-cleaner.py ${DATA_DIR} --ann ${ANN_DIR}
#/miniconda/envs/vmr/bin/python

