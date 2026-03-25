#!/bin/bash
#BSUB -J prepare_data_unmatched
#BSUB -o /zhome/d0/a/221493/thesis/logs/prepare_data_unmatched%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/prepare_data_unmatched%J.err
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 3:00
#BSUB -u s251710@dtu.dk
#BSUB -B
#BSUB -N

mkdir -p /zhome/d0/a/221493/thesis/logs
source /work3/s251710/thesis_env/bin/activate
python /zhome/d0/a/221493/thesis/code/c2_prepare_unmatched_cf.py