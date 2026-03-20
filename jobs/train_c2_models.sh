#!/bin/bash
#BSUB -J train_C2_models_CV
#BSUB -o /zhome/d0/a/221493/thesis/logs/train_C2_models_CV%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/train_C2_models_CV%J.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u s251710@dtu.dk
#BSUB -B
#BSUB -N

mkdir -p /zhome/d0/a/221493/thesis/logs
source /work3/s251710/thesis_env/bin/activate
python /zhome/d0/a/221493/thesis/code/c2_train_simulated_cf.py