#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train CNN to predict DICOM acquisition parameters from X-rays.
Designed for HPC GPU job submission.
"""

import os
import argparse
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================
# Dataset
# =========================

class XrayAcquisitionDataset(Dataset):
    def __init__(self, dcm_root, transform=None):
        self.transform = transform
        self.samples = []

        for patient in os.listdir(dcm_root):
            patient_path = os.path.join(dcm_root, patient)
            if not os.path.isdir(patient_path):
                continue
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue
                for dcm_file in os.listdir(study_path):
                    if dcm_file.endswith(".dcm"):
                        self.samples.append(os.path.join(study_path, dcm_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path = self.samples[idx]
        ds = pydicom.dcmread(dcm_path)

        # Image
        img_array = ds.pixel_array.astype(np.float32)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        img = Image.fromarray((img_array * 255).astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        # Safe DICOM extraction
        def get_dicom_value(ds, tag, default=0.0):
            val = ds.get(tag, None)
            if val is None:
                return float(default)
            return float(val.value)

        target = torch.tensor([
            get_dicom_value(ds, (0x0018, 0x0060)),  # kVp
            get_dicom_value(ds, (0x0018, 0x1152)),  # mAs
            get_dicom_value(ds, (0x0018, 0x1150))   # exposure time
        ], dtype=torch.float32)

        # Normalize targets (important!)
        target[0] /= 150.0
        target[1] /= 50.0
        target[2] /= 1000.0

        return img, target


# =========================
# Training
# =========================

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    train_dataset = XrayAcquisitionDataset(args.train_dir, transform=transform)
    val_dataset   = XrayAcquisitionDataset(args.val_dir, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # =========================
    # Training Loop
    # =========================

    for epoch in range(args.epochs):

        model.train()
        train_loss = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs} "
              f"- Train Loss: {train_loss:.4f} "
              f"- Val Loss: {val_loss:.4f}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "acquisition_model.pth"))

    print("Training complete. Model saved.")


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    train(args)