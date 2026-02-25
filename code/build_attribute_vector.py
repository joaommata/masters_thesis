# scripts/build_attribute_vector_refactored.py
# João Mata 16-02-2026 (NPC-ready)

import os
import logging
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from radiomics import featureextractor
import SimpleITK as sitk
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Headless mode for NPC
import matplotlib.pyplot as plt

# Suppress radiomics warnings
logger = logging.getLogger('radiomics')
logger.setLevel(logging.ERROR)

class FeatureVectorBuilder:
    def __init__(self, models, segmentation_model=None, radiomics_extractor=None, device=None):
        self.models = models
        self.segmentation_model = segmentation_model
        self.extractor = radiomics_extractor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in self.models.values():
            model.to(self.device).eval()
        if self.segmentation_model:
            self.segmentation_model.to(self.device).eval()

    # --------------------
    # Model Predictions
    # --------------------
    def predict_age(self, img_tensor):
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.models['age'](img_tensor).cpu().numpy()[0][0]
        return float(pred)

    def predict_sex(self, img_tensor):
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.models['sex'](img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx_male = self.models['sex'].targets.index("Male")
        idx_female = self.models['sex'].targets.index("Female")
        return probs[idx_male], probs[idx_female]

    def predict_race(self, img_tensor):
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.models['race'](img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx_white = self.models['race'].targets.index("White")
        idx_black = self.models['race'].targets.index("Black")
        idx_asian = self.models['race'].targets.index("Asian")
        return probs[idx_white], probs[idx_black], probs[idx_asian]

    # --------------------
    # Radiomics Features
    # --------------------
    def extract_radiomics_features(self, img_np, mask_np, class_name):
        mask_resized = cv2.resize(
            mask_np,
            (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        mask_resized = (mask_resized > 0).astype(np.uint8)

        if mask_resized.sum() == 0:
            print(f"[SKIP] {class_name} mask is empty.")
            return {}
        if mask_resized.shape[0] < 2 or mask_resized.shape[1] < 2:
            print(f"[SKIP] {class_name} mask too small: {mask_resized.shape}")
            return {}

        img_sitk = sitk.GetImageFromArray(img_np.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask_resized.astype(np.uint8))

        try:
            feats = self.extractor.execute(img_sitk, mask_sitk)
            feats = {f"{class_name}_{k}": v for k, v in feats.items() if not k.startswith("diagnostics_")}
        except Exception as e:
            print(f"[ERROR] Failed to extract features for {class_name}: {e}")
            feats = {}

        return feats

    # --------------------
    # Plotting (optional)
    # --------------------
    def plot_segmentation(self, img_np, seg_output, save_path=None):
        num_classes = len(self.segmentation_model.targets)
        plt.figure(figsize=(26, 5))
        plt.subplot(1, num_classes + 1, 1)
        plt.imshow(img_np, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        for i in range(num_classes):
            plt.subplot(1, num_classes + 1, i + 2)
            plt.imshow(seg_output[0, i], cmap='viridis')
            plt.title(self.segmentation_model.targets[i])
            plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    # --------------------
    # Build full vector
    # --------------------
    def build_vector(self, img_tensor, img_np, img_path, plot=False):
        parts = img_path.replace("\\", "/").split("/")

        age_pred = self.predict_age(img_tensor)
        sex_male, sex_female = self.predict_sex(img_tensor)
        race_white, race_black, race_asian = self.predict_race(img_tensor)

        vector = {
            'img_id': parts[-3] if len(parts) >= 3 else parts[-2],
            'path': img_path,
            'age_pred': age_pred,
            'sex_male': sex_male,
            'sex_female': sex_female,
            'race_white': race_white,
            'race_black': race_black,
            'race_asian': race_asian
        }

        if self.segmentation_model:
            with torch.no_grad():
                seg_output = self.segmentation_model(img_tensor.unsqueeze(0).to(self.device)).cpu().numpy()
            seg_output = 1 / (1 + np.exp(-seg_output))
            seg_output = (seg_output >= 0.5).astype(np.uint8)

            if plot:
                self.plot_segmentation(img_np, seg_output)

            for class_idx, class_name in enumerate(self.segmentation_model.targets):
                mask_np = seg_output[0, class_idx]
                if mask_np.sum() == 0:
                    continue
                rad_feats = self.extract_radiomics_features(img_np, mask_np, class_name)
                vector.update(rad_feats)

        return vector


def main():
    # Base paths relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "CheXpert-v1.0-small")
    csv_path = os.path.join(data_path, "valid.csv")
    output_path = os.path.join(base_dir, "results", "attribute_vectors.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading Models and Dataset...")

    models = {
        'age': xrv.baseline_models.riken.AgeModel(),
        'sex': xrv.baseline_models.mira.SexModel(),
        'race': xrv.baseline_models.emory_hiti.RaceModel()
    }

    segmentation_model = xrv.baseline_models.chestx_det.PSPNet()
    radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)

    builder = FeatureVectorBuilder(
        models=models,
        segmentation_model=segmentation_model,
        radiomics_extractor=radiomics_extractor,
    )

    transform = xrv.datasets.XRayResizer(224)
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=data_path,
        csvpath=csv_path,
        views=["PA", "AP"],
        transform=transform
    )

    print("Building feature vectors...")
    all_vectors = []
    for i in tqdm(range(len(dataset)), desc="Building feature vectors"):
        sample = dataset[i]
        row = dataset.csv.iloc[i]
        img_path = row["Path"]

        img_array = sample['img']
        if len(img_array.shape) == 2:  # (H,W)
            img_np = img_array
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
        else:  # (C,H,W)
            img_np = img_array[0]
            img_tensor = torch.from_numpy(img_array).float()

        vector = builder.build_vector(img_tensor, img_np, img_path=img_path, plot=False)
        all_vectors.append(vector)

    df = pd.DataFrame(all_vectors)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(all_vectors)} feature vectors to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()