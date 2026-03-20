#!/bin/bash
#BSUB -J build_attr_vectors_rad
#BSUB -o /zhome/d0/a/221493/thesis/logs/build_attr_rad_%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/build_attr_rad_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 12:00
#BSUB -u s251710@dtu.dk
#BSUB -N

mkdir -p /zhome/d0/a/221493/thesis/logs

source /work3/s251710/thesis_env/bin/activate

python /zhome/d0/a/221493/thesis/code/c1_build_attribute_vector.py