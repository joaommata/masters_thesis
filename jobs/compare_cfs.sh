#!/bin/bash
#BSUB -J c2_multiple_cf
#BSUB -o /zhome/d0/a/221493/thesis/logs/c2_multiple_cf%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/c2_multiple_cf%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 0:30
#BSUB -u s251710@dtu.dk
#BSUB -B
#BSUB -N

mkdir -p /zhome/d0/a/221493/thesis/logs
source /work3/s251710/thesis_env/bin/activate
python /zhome/d0/a/221493/thesis/code/c2_multiple_cf.py