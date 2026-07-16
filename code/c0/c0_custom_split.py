# %% [markdown]
# ## C0 classifier - new custom split
# #### For effusion prediction (homemade version) -> current version is for pneumothorax
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
DATA_ROOT   = os.environ.get("THESIS_DATA", "/work3/s251710/thesis_data")
DATA_DIR = DATA_ROOT + "/"
DISEASE = 'pneumothorax'

# Take the original train set
original_train = pd.read_csv(os.path.join(DATA_DIR, "CheXpert-v1.0-small/train.csv"))
print(f"Original train set shape: {original_train.shape}")

# CHANGED TO PNEUMOTHORAX
# disease col is "Pneumothorax" for pneumothorax prediction, and "Pleural Effusion" for effusion prediction
# disease to column mapping
disease_col = {
    "pneumothorax": "Pneumothorax",
    "effusion": "Pleural Effusion"
}

# Drop samples with -1 (uncertainty or Nan, before splitting into train and validation sets
original_train = original_train.dropna(subset=[disease_col[DISEASE]])
original_train = original_train[original_train[disease_col[DISEASE]].isin([0.0, 1.0])]
original_train = original_train.reset_index(drop=True)

# Patient-level stratified split - Each patient has multiple images, so we split by patient ID not by image
# Extract patient ID from path (structure is: .../patientXXXXX/...)
original_train["patient_id"] = original_train["Path"].apply(lambda x: x.split("/")[2])
print(original_train["Path"].iloc[0])  # Example path to check patient ID extraction
print(original_train["patient_id"].iloc[0])  # Check extracted patient ID

# Get unique patients and their majority pneumothorax label for stratification
patient_labels = (
    original_train.groupby("patient_id")[disease_col[DISEASE]]
    .agg(lambda x: x.mode()[0])  # most common label per patient
    .reset_index()
)

# Split at patient level
train_patients, val_patients = train_test_split(
    patient_labels,
    test_size=0.5,
    random_state=42,
    stratify=patient_labels[disease_col[DISEASE]]
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
# Look at the distribution of the target variable (Pneumothorax) in the training set
train_pneumothorax_count = train_df[disease_col[DISEASE]].value_counts()
print("Pneumothorax distribution in training set:")
print(train_pneumothorax_count)

valid_pneumothorax_count = val_df[disease_col[DISEASE]].value_counts()
print("Pneumothorax distribution in validation set:")
print(valid_pneumothorax_count)

# Plot the distribution of the target variable in the training set and validation set (different colors)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
train_pneumothorax_count.plot(kind="bar", color=["blue", "orange", "green"])
plt.title(f"{DISEASE.capitalize()} Distribution in Training Set")
plt.xlabel(f"{DISEASE.capitalize()} Label")
plt.ylabel("Count")
plt.subplot(1, 2, 2)
valid_pneumothorax_count.plot(kind="bar", color=["blue", "orange", "green"])
plt.title(f"{DISEASE.capitalize()} Distribution in Validation Set")
plt.xlabel(f"{DISEASE.capitalize()} Label")
plt.ylabel("Count")
plt.tight_layout()
# Save the plot
plt.show()


# Visualize sample images from the training set for both classes (Pneumothorax = 0 and Pneumothorax = 1)
pneumothorax_0_samples = train_df[train_df[disease_col[DISEASE]] == 0].sample(5, random_state=42)
pneumothorax_1_samples = train_df[train_df[disease_col[DISEASE]] == 1].sample(5, random_state=42)
plt.figure(figsize=(15, 6))
for i, (index, row) in enumerate(pneumothorax_0_samples.iterrows()):
    img_path = os.path.join(DATA_DIR, row["Path"])
    img = plt.imread(img_path)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{DISEASE.capitalize()} = 0")
    plt.axis("off")
for i, (index, row) in enumerate(pneumothorax_1_samples.iterrows()):
    img_path = os.path.join(DATA_DIR, row["Path"])
    img = plt.imread(img_path)
    plt.subplot(2, 5, i + 6)
    plt.imshow(img, cmap="gray")
    plt.title(f"{DISEASE.capitalize()} = 1")
    plt.axis("off")
plt.tight_layout()
# Save the plot
plt.savefig(os.path.join(DATA_DIR, f"{DISEASE}_sample_images.png"))
plt.show()


# Print size of both training and validation sets
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")

# Save as csv file 
train_path = os.path.join(DATA_DIR, f"{DISEASE}/custom_train_split.csv")
val_path = os.path.join(DATA_DIR, f"{DISEASE}/custom_val_split.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

