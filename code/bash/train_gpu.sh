#!/bin/bash
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 4
#BSUB -J acq_train
#BSUB -o /zhome/d0/a/221493/thesis/logs/%J.out
#BSUB -e /zhome/d0/a/221493/thesis/logs/%J.err

source /zhome/d0/a/221493/thesis/venv/bin/activate

python /zhome/d0/a/221493/thesis/code/train_acquisition_model.py \
    --train_dir /zhome/d0/a/221493/thesis/data/dcm_chexpert_plus_chunk_0/train \
    --val_dir   /zhome/d0/a/221493/thesis/data/dcm_chexpert_plus_chunk_0/valid \
    --epochs 10 \
    --batch_size 16 \
    --output_dir /zhome/d0/a/221493/thesis/output