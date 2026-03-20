#!/bin/bash
#BSUB -J c1_extend_radiomics
#BSUB -o /zhome/d0/a/221493/thesis/logs/c1_extend_radiomics_%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/c1_extend_radiomics_%J.err
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s251710@dtu.dk
#BSUB -B
#BSUB -N

mkdir -p /zhome/d0/a/221493/thesis/logs
source /work3/s251710/thesis_env/bin/activate
python /zhome/d0/a/221493/thesis/code/c1_extend_radiomics.py