#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

# Import your dataset class from training file
from train_acquisition_model import XrayAcquisitionDataset


def evaluate(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Same transforms as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    dataset = XrayAcquisitionDataset(args.val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Validation samples:", len(dataset))

    # Recreate model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * imgs.size(0)

    total_loss /= len(loader.dataset)

    print(f"Validation MSE: {total_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    evaluate(args)
    
