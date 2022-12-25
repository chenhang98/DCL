#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PROJ_ROOT=`pwd`
export PYTHONPATH=$PROJ_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

WORK_DIR=$3

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJ_ROOT/tools/train.py $CONFIG --work-dir=${WORK_DIR} --seed 0 --launcher pytorch ${@:4}
