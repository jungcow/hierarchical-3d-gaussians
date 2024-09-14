#!/bin/bash

#SBATCH -A jw@3090
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00

export CUDA_VISIBLE_DEVICES=2,3


COARSE_ENV=$1
COLMAP_DIR=$2
IMAGES_DIR=$3
OUTPUT_DIR=$4

if [ -z "${5}" ]; then
    echo "No masks provided."
    MASKS_ARG=""
else
    MASKS_ARG="${5}"
    echo "masks provided: $MASKS_ARGS"
fi

source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate ${COARSE_ENV}

python ..//train_coarse.py -s ${COLMAP_DIR} --save_iterations -1 -i ${IMAGES_DIR} --skybox_num 100000 --model_path ${OUTPUT_DIR}/scaffold ${MASKS_ARGS} 
