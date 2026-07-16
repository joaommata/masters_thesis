# scripts/c1_build_attribute_vector.py
# João Mata 16-02-2026

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchxrayvision as xrv
from radiomics import featureextractor
import SimpleITK as sitk
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Headless mode for NPC
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Suppress radiomics warnings they were annoying
logger = logging.getLogger('radiomics')
logger.setLevel(logging.ERROR)


class FeatureVectorBuilder:
    def __init__(self, models, segmentation_model=None, radiomics_extractor=None, device=None):
        self.models = models
        self.segmentation_model = segmentation_model
        self.extractor = radiomics_extractor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move models to device and set to eval mode to ensure they don't update their weights and to speed up inference
        for model in self.models.values():
            model.to(self.device).eval()
        if self.segmentation_model:
            self.segmentation_model.to(self.device).eval()

    # --------------------
    # Batched Model Predictions -> OPTIMIZED using AI to reduce GPU overhead and speed up processing
    # --------------------
    def predict_age_batch(self, img_batch):
        img_batch = img_batch.to(self.device)
        with torch.no_grad():
            preds = self.models['age'](img_batch).cpu().numpy()
        return [float(p[0]) for p in preds]

    def predict_sex_batch(self, img_batch):
        img_batch = img_batch.to(self.device)
        with torch.no_grad():
            logits = self.models['sex'](img_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        idx_male   = self.models['sex'].targets.index("Male")
        idx_female = self.models['sex'].targets.index("Female")
        return [(probs[i][idx_male], probs[i][idx_female]) for i in range(len(probs))]

    def predict_race_batch(self, img_batch):
        img_batch = img_batch.to(self.device)
        with torch.no_grad():
            logits = self.models['race'](img_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        idx_white = self.models['race'].targets.index("White")
        idx_black = self.models['race'].targets.index("Black")
        idx_asian = self.models['race'].targets.index("Asian")
        return [(probs[i][idx_white], probs[i][idx_black], probs[i][idx_asian]) for i in range(len(probs))]

    def predict_segmentation_batch(self, img_batch):
        img_batch = img_batch.to(self.device)
        with torch.no_grad():
            seg_output = self.segmentation_model(img_batch).cpu().numpy()
        seg_output = 1 / (1 + np.exp(-seg_output))
        seg_output = (seg_output >= 0.5).astype(np.uint8)
        return seg_output  # (B, num_classes, H, W)

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

        # Safeguards against empty or too small masks which can cause radiomics extraction to fail. 
        # This can happen if the segmentation model fails to detect the class in the image.
        if mask_resized.ndim != 2:
            return {}
        if mask_resized.sum() < 10:
            print(f"[SKIP] {class_name} mask is empty or too small.")
            return {}
        ys, xs = np.where(mask_resized > 0)
        if (ys.max() - ys.min()) < 2 or (xs.max() - xs.min()) < 2:
            print(f"[SKIP] {class_name} mask has no 2D extent.")
            return {}
        
        # Convert to SimpleITK images for radiomics. Radiomics expects the image and mask to be in a specific format, so we need to convert our numpy arrays to SimpleITK images. 
        img_sitk  = sitk.GetImageFromArray(img_np.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask_resized.astype(np.uint8))

        # Extract features using the radiomics extractor.
        try:
            feats = self.extractor.execute(img_sitk, mask_sitk)
            feats = {f"{class_name}_{k}": v for k, v in feats.items() if not k.startswith("diagnostics_")}
        
        # This can fail for various reasons (e.g. if the mask is not valid, if the image has too few pixels, etc.) so we wrap it in a try-except block to catch any errors and continue processing other classes/images.
        except Exception as e:
            print(f"[ERROR] Failed to extract features for {class_name}: {e}")
            feats = {}

        return feats

    # --------------------
    # Interpretable Geometric Features 
    # - Height, width, area, perimeter, etc.
    # - For now these can serve as a proxy for radiomics features to speed up processing, but we can always add more later if needed. 
    # - These features are also more interpretable and can be useful for analysis and understanding model behavior.
    # --------------------
    def extract_geometry_features(self, mask_np, class_name, pixel_spacing=None):
        geo_feats = {}

        area_pixels = float(mask_np.sum())
        geo_feats[f"{class_name}_area_pixels"] = area_pixels

        ys, xs = np.where(mask_np > 0)
        if len(xs) > 0 and len(ys) > 0:
            width  = xs.max() - xs.min()
            height = ys.max() - ys.min()
            geo_feats[f"{class_name}_bbox_width"]  = float(width)
            geo_feats[f"{class_name}_bbox_height"] = float(height)
            geo_feats[f"{class_name}_bbox_ratio"]  = float(width / (height + 1e-6))

        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            geo_feats[f"{class_name}_perimeter"] = float(perimeter)

        return geo_feats

    # --------------------
    # Plotting (optional) - can be useful for debugging and visualization, but can be disabled for speed when processing large batches.
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
    # Build vectors for a full batch
    # --------------------
    def build_vectors_batch(self, img_tensors, img_nps, img_paths, plot=False):
        B = img_tensors.shape[0]

        age_preds  = self.predict_age_batch(img_tensors)
        sex_preds  = self.predict_sex_batch(img_tensors)
        race_preds = self.predict_race_batch(img_tensors)

        seg_outputs = None
        if self.segmentation_model:
            seg_outputs = self.predict_segmentation_batch(img_tensors)

        # Build initial vectors with model predictions
        vectors = []

        for b in range(B):
            parts = img_paths[b].replace("\\", "/").split("/")
            vector = {
                'patient_id': parts[-3] if len(parts) >= 3 else parts[-2],
                'path':       img_paths[b],
                'age_pred':   age_preds[b],
                'sex_male':   sex_preds[b][0],
                'sex_female': sex_preds[b][1],
                'race_white': race_preds[b][0],
                'race_black': race_preds[b][1],
                'race_asian': race_preds[b][2],
            }
            vectors.append(vector)

        # If i have a segmentation output, i want to extract radiomics features for each class in parallel
        if seg_outputs is not None:
            # Build a flat list of all (b, class_name, img_np, mask_np) tasks across the whole batch
            tasks = []
            for b in range(B):
                for class_idx, class_name in enumerate(self.segmentation_model.targets):
                    mask_np = seg_outputs[b, class_idx]
                    if mask_np.sum() == 0:
                        continue
                    tasks.append((b, class_name, img_nps[b], mask_np))

            def run_radiomics(args):
                b, class_name, img_np, mask_np = args
                
                ## WE ARE MAKING A SMALLER SIMPLER VERSION FIRST TO SPEED UP PROCESSING AND TEST THE PIPELINE. THIS CAN BE EXPANDED LATER TO INCLUDE MORE FEATURES IF NEEDED.
                
                # Geometry on everything - it's essentially free
                geom_feats = self.extract_geometry_features(mask_np, class_name)
                
                rad_feats = self.extract_radiomics_features(img_np, mask_np, class_name)

                return b, {**geom_feats, **rad_feats}

            with ThreadPoolExecutor(max_workers=8) as executor:
                for b, feats in executor.map(run_radiomics, tasks):
                    vectors[b].update(feats)

        return vectors


def build_split(csv_filename, output_filename, base_dir, models, segmentation_model, radiomics_extractor,
                batch_size=16, num_workers=4):
    # Set up paths
    data_path   = os.path.join(base_dir, "data", "CheXpert-v1.0-small")
    csv_path    = os.path.join(data_path, csv_filename)
    output_path = os.path.join(base_dir, "results/C1_attributes", output_filename)

    print(f"\nProcessing split: {csv_filename}")
    print(f"Saving to: {output_path}")
    print(f"Batch size: {batch_size} | DataLoader workers: {num_workers}")

    # Initialize the feature vector builder with the models and extractor
    builder = FeatureVectorBuilder(
        models=models,
        segmentation_model=segmentation_model,
        radiomics_extractor=radiomics_extractor,
    )
    # Use the same resizing transform as the models expect
    transform = xrv.datasets.XRayResizer(224)

    # Create dataset and dataloader
    dataset = xrv.datasets.CheX_Dataset(
        imgpath=data_path,
        csvpath=csv_path,
        views=["PA", "AP"],
        transform=transform,
        unique_patients=False   # include all samples, even if they are from the same patient, to maximize data for C1
    )
    
    print("Dataset samples (len(dataset)):", len(dataset))
    print("Internal CSV length:", len(dataset.csv))

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    all_vectors = []
    start_index = 0
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        if len(df_existing) > 0:
            already_done = set(df_existing["path"].tolist())
            all_vectors  = df_existing.to_dict("records")
            # Find the highest sequential index we can safely resume from
            # (last path in dataset order that is in the checkpoint)
            dataset_paths = dataset.csv["Path"].tolist()
            for i, p in enumerate(dataset_paths):
                if p in already_done:
                    start_index = i + 1
                else:
                    break
            print(f"[RESUME] Found checkpoint with {len(df_existing):,} samples.")
            print(f"[RESUME] Resuming from dataset index {start_index:,} (skipping first {start_index:,} images).")
    else:
        print("[INFO] No checkpoint found, starting from scratch.")

    # Subset the dataset to only unprocessed samples
    remaining_indices = list(range(start_index, len(dataset)))
    subset = torch.utils.data.Subset(dataset, remaining_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    samples_processed = start_index  # keep absolute index for path lookup

    for batch in tqdm(loader, desc=f"Building {csv_filename}"):
        try:
            # Details of batch processing, I don't fully understand
            img_tensors = batch['img'].float()  # (B, C, H, W)
            B           = img_tensors.shape[0]
            img_paths   = dataset.csv.iloc[samples_processed:samples_processed + B]['Path'].tolist()
            img_nps     = [img_tensors[b, 0].numpy() for b in range(B)]

            vectors = builder.build_vectors_batch(img_tensors, img_nps, img_paths, plot=False)
            all_vectors.extend(vectors)

        except Exception as e:
            print(f"[SKIP] Failed to process batch starting at index {samples_processed}: {e}")

        finally:
            samples_processed += img_tensors.shape[0]

        # Save checkpoint every ~500 samples to avoid losing progress and to monitor intermediate results
        if samples_processed % 500 < batch_size:
            df_temp = pd.DataFrame(all_vectors)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_temp.to_csv(output_path, index=False)
            print(f"[INFO] Saved {len(df_temp)} samples so far to {output_path}")
        
    # Final save after all batches are processed
    df = pd.DataFrame(all_vectors)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    print(f"Total features: {df.shape[1]}")


def main():
    # Determine base directory (one level up from this script)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Base directory: {base_dir}")

    print("Loading models...")
    # Load the models from torchxrayvision. These are pretrained on CheXpert and will be used to predict age
    models = {
        'age':  xrv.baseline_models.riken.AgeModel(),
        'sex':  xrv.baseline_models.mira.SexModel(),
        'race': xrv.baseline_models.emory_hiti.RaceModel()
    }
    segmentation_model  = xrv.baseline_models.chestx_det.PSPNet()
    radiomics_extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    radiomics_extractor.disableAllFeatures()
    radiomics_extractor.enableFeatureClassByName('firstorder')
    radiomics_extractor.enableFeatureClassByName('shape2D')

    print("Building attribute vectors for training split...")
    # First for the training split, we will build the attribute vectors by running the images through the models and extracting features. This will be saved to a new CSV file that will be used in the next steps of the pipeline.
    build_split(
        csv_filename="train.csv",
        output_filename="train_c1_attribute_vector_rad.csv",
        base_dir=base_dir,
        models=models,
        segmentation_model=segmentation_model,
        radiomics_extractor=radiomics_extractor,
        batch_size=16,   # increase to 32/64 if GPU VRAM allows
        num_workers=4,   # set to your CPU core count
    )

    # Then we will do the same for the validation split. This will allow us to have attribute vectors for both training and validation data, which can be used for analysis and model development in the next steps.
    print("Building attribute vectors for validation split...")
    build_split(
        csv_filename="valid.csv",
        output_filename="valid_c1_attribute_vector_rad.csv",
        base_dir=base_dir,
        models=models,
        segmentation_model=segmentation_model,
        radiomics_extractor=radiomics_extractor,
        batch_size=16,
        num_workers=4,
    )

if __name__ == "__main__":
    main()