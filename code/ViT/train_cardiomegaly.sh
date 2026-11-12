#!/bin/bash
#BSUB -J train_c0_cardiomegaly_vit
#BSUB -o /zhome/d0/a/221493/thesis/logs/train_c0_cardiomegaly_vit_%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/train_c0_cardiomegaly_vit_%J.err
#BSUB -q gpul40s
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 4:00
#BSUB -u s251710@dtu.dk
#BSUB -B
#BSUB -N
# Ensure the log directory exists
mkdir -p /zhome/d0/a/221493/thesis/logs

# Activate your virtual environment
source /work3/s251710/thesis_env/bin/activate
source /zhome/d0/a/221493/thesis/jobs/stage_chexpert.sh   # stage CheXpert to node-local disk; sets THESIS_DATA

# Execute the python script with the hardcoded target
python /zhome/d0/a/221493/thesis/code/ViT/c0_train_vit.py --disease cardiomegaly