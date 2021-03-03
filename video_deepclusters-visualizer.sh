# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

PYTHON="/home/${USER}/miniconda/envs/vmr/bin/python"
CLUSTERS="/thecube/students/${USER}/test/exp/test/clusters"
DATASET="/mnt/ILSVRC2017_VID/ILSVRC/Data/VID/train/"
SAVE_DIR="/thecube/students/${USER}/cluster_images"
IMG_NUM="9"

mkdir -p ${SAVE_DIR}
${PYTHON} clusters-visualizer.py ${CLUSTERS} ${DATASET} --save_dir ${SAVE_DIR} --img_num ${IMG_NUM}
