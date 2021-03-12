#!/bin/bash

export PYTHONPATH="`pwd`:${PYTHONPATH}"
if [ $# != 5 ]
then 
  echo "Please specify 1) cfg; 2) gpus; 3) method; 4) exp_name; 5) clustering_wt."
  exit
fi

cfg=${1}
gpus=${2}
method=${3}
exp_name=${4}
clustering_wt=${5}

out_dir=./experiments/ckpt/${exp_name}
if [ -d ${out_dir} ]
then
  rm -rf ${out_dir}
fi
mkdir -p ${out_dir}

CUDA_VISIBLE_DEVICES=${gpus} python3 ./tools/train.py --cfg ${cfg} \
           --method ${3} --exp_name ${4} --clustering_wt ${5} 2>&1 | tee ${out_dir}/log.txt
