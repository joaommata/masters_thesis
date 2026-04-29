# %% [markdown]
# ## C0 classifier - new custom split
# #### For effusion prediction (homemade version)

# %%
import pandas as pd
import os
import matplotlib.pyplot as plt

# %%
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# SETUP
DATA_DIR = "/zhome/d0/a/221493/thesis/data/"

# Take the original train set
original_train = pd.read_csv(os.path.join(DATA_DIR, "CheXpert-v1.0-small/train.csv"))
print(f"Original train set shape: {original_train.shape}")


# Drop samples with -1 (uncertainty or Nan, before splitting into train and validation sets
original_train = original_train.dropna(subset=["Pleural Effusion"])
original_train = original_train[original_train["Pleural Effusion"].isin([0.0, 1.0])]
original_train = original_train.reset_index(drop=True)

# Patient-level stratified split - Each patient has multiple images, so we split by patient ID not by image
# Extract patient ID from path (structure is: .../patientXXXXX/...)
original_train["patient_id"] = original_train["Path"].apply(lambda x: x.split("/")[2])
print(original_train["Path"].iloc[0])  # Example path to check patient ID extraction
print(original_train["patient_id"].iloc[0])  # Check extracted patient ID

# Get unique patients and their majority effusion label for stratification
patient_labels = (
    original_train.groupby("patient_id")["Pleural Effusion"]
    .agg(lambda x: x.mode()[0])  # most common label per patient
    .reset_index()
)

# Split at patient level
train_patients, val_patients = train_test_split(
    patient_labels,
    test_size=0.5,
    random_state=42,
    stratify=patient_labels["Pleural Effusion"]
)

# Check if patients overlap between train and validation sets (should be zero)
overlap = set(train_patients["patient_id"]).intersection(set(val_patients["patient_id"]))
print(f"Overlap in patient IDs between train and validation sets: {len(overlap)}")

# Map back to image rows
train_df = original_train[original_train["patient_id"].isin(train_patients["patient_id"])]
val_df   = original_train[original_train["patient_id"].isin(val_patients["patient_id"])]
print(f"Train set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")

# %%
# Look at the distribution of the target variable (Effusion) in the training set
train_effusion_count = train_df["Pleural Effusion"].value_counts()
print("Effusion distribution in training set:")
print(train_effusion_count)

valid_effusion_count = val_df["Pleural Effusion"].value_counts()
print("Effusion distribution in validation set:")
print(valid_effusion_count)

# Plot the distribution of the target variable in the training set and validation set (different colors)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
train_effusion_count.plot(kind="bar", color=["blue", "orange", "green"])
plt.title("Effusion Distribution in Training Set")
plt.xlabel("Effusion Label")
plt.ylabel("Count")
plt.subplot(1, 2, 2)
valid_effusion_count.plot(kind="bar", color=["blue", "orange", "green"])
plt.title("Effusion Distribution in Validation Set")
plt.xlabel("Effusion Label")
plt.ylabel("Count")
plt.tight_layout()
# Save the plot
plt.show()


# Visualize sample images from the training set for both classes (Effusion = 0 and Effusion = 1)
effusion_0_samples = train_df[train_df["Pleural Effusion"] == 0].sample(5, random_state=42)
effusion_1_samples = train_df[train_df["Pleural Effusion"] == 1].sample(5, random_state=42)
plt.figure(figsize=(15, 6))
for i, (index, row) in enumerate(effusion_0_samples.iterrows()):
    img_path = os.path.join(DATA_DIR, row["Path"])
    img = plt.imread(img_path)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title("Effusion = 0")
    plt.axis("off")
for i, (index, row) in enumerate(effusion_1_samples.iterrows()):
    img_path = os.path.join(DATA_DIR, row["Path"])
    img = plt.imread(img_path)
    plt.subplot(2, 5, i + 6)
    plt.imshow(img, cmap="gray")
    plt.title("Effusion = 1")
    plt.axis("off")
plt.tight_layout()
plt.show()


# Print size of both training and validation sets
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Save as csv file 
train_path = os.path.join(DATA_DIR, "custom_train_split.csv")
val_path = os.path.join(DATA_DIR, "custom_val_split.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

