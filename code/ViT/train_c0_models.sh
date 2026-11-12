#!/bin/bash
#BSUB -J train_c0_effusion_vit
#BSUB -o /zhome/d0/a/221493/thesis/logs/train_c0_effusion_vit_%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/train_c0_effusion_vit_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 4:00
#BSUB -u s251710@dtu.dk
#BSUB -B
#BSUB -N

mkdir -p /zhome/d0/a/221493/thesis/logs
source /work3/s251710/thesis_env/bin/activate
source /zhome/d0/a/221493/thesis/jobs/stage_chexpert.sh   # stage CheXpert to node-local disk; sets THESIS_DATA
python /zhome/d0/a/221493/thesis/code/ViT/c0_train_vit.py