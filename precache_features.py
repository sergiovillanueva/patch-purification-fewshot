"""
Pre-cache features for all datasets (DINOv3 or CLIP backbone).
================================================================
Extracts features ONCE for all images (train + test) and saves to disk as .npy files.
After this, the experiment scripts can run without GPU (pure CPU: LOO + KNN).

Output structure:
  output/feature_cache/{dataset_safe_name}/         # DINOv3 features
    all_train_features.npy   # (n_all_train, 784, 1024)
    all_train_paths.txt
    test_features.npy         # (n_test, 784, 1024)
    test_paths.txt
    test_labels.npy

  output/feature_cache_clip/{dataset_safe_name}/    # CLIP features
    all_train_features.npy   # (n_all_train, 256, 1024)
    ...

Usage:
  uv run precache_features.py                      # DINOv3, all 35 datasets
  uv run precache_features.py --backbone clip      # CLIP ViT-L/14, all 35 datasets
  uv run precache_features.py --debug              # 2 datasets only
  uv run precache_features.py --backbone clip --debug
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import transformers
from transformers import AutoModel
transformers.logging.set_verbosity_error()

# =============================================================================
# CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser(description="Pre-cache features for all datasets")
parser.add_argument("--debug", action="store_true", help="Only 2 datasets")
parser.add_argument("--backbone", choices=["dino", "clip", "dinov2"], default="dino",
                    help="Backbone: dino (DINOv3-ViT-L, 448×448, 784 patches), "
                         "clip (CLIP ViT-L/14, 224×224, 256 patches) or "
                         "dinov2 (DINOv2-ViT-L/14, 448×448, 1024 patches)")
_ARGS = parser.parse_args()
DEBUG = _ARGS.debug
BACKBONE = _ARGS.backbone

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data")

# Backbone-specific config
if BACKBONE == "dino":
    CACHE_DIR = Path("output/feature_cache")
    RESOLUTION = 448
    LAYER_IDX = -6
    BATCH_SIZE = 16
elif BACKBONE == "clip":
    CACHE_DIR = Path("output/feature_cache_clip")
    RESOLUTION = 224
    LAYER_IDX = -6  # Use intermediate layer for richer features (like DINOv3)
    BATCH_SIZE = 32  # CLIP is smaller, can use larger batch
elif BACKBONE == "dinov2":
    CACHE_DIR = Path("output/feature_cache_dinov2")
    RESOLUTION = 448  # 448 / 14 = 32 -> 1024 patches
    LAYER_IDX = -6
    BATCH_SIZE = 12  # 1024 patch tokens per image, keep VRAM in check

# 35 datasets (MVTec AD + VisA + BTAD + MVTec LOCO AD)
ALL_DATASETS = [
    # MVTec AD (15)
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
    # VisA (12)
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum", "VisA/fryum",
    "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1", "VisA/pcb2", "VisA/pcb3",
    "VisA/pcb4", "VisA/pipe_fryum",
    # MVTec LOCO AD (5)
    "mvtec_loco_AD/breakfast_box", "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins", "mvtec_loco_AD/screw_bag",
    "mvtec_loco_AD/splicing_connectors",
    # BTAD (3)
    "btad/01", "btad/02", "btad/03",
]


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class DINOv3Extractor:
    def __init__(self, device: str = "cuda",
                 model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
                 patch_size: int = 16,
                 num_register: int | None = None):
        """Also used for DINOv2 (patch_size=14, num_register=0: DINOv2 has no
        register tokens, so the config-based default of 4 would silently drop
        four real patch tokens)."""
        self.device = device
        print(f"[FEAT] Loading {model_name} on {device}...")
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        self.patch_size = patch_size
        self.grid_size = RESOLUTION // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        if num_register is not None:
            self.num_register = num_register
        else:
            self.num_register = getattr(self.model.config, "num_register_tokens", 4)
        self.feature_dim = self.model.config.hidden_size

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[FEAT] Grid: {self.grid_size}x{self.grid_size}, patches: {self.num_patches}, dim: {self.feature_dim}")

    @torch.no_grad()
    def extract_batch(self, image_paths: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_features = []
        start_idx = 1 + self.num_register

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(self.transform(img))
                img.close()
            batch = torch.stack(images).to(self.device)

            with torch.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
                outputs = self.model(batch, output_hidden_states=True)

            features = outputs.hidden_states[LAYER_IDX][:, start_idx:start_idx + self.num_patches, :]
            all_features.append(features.float().cpu().numpy())

            del batch, outputs, features

        return np.concatenate(all_features, axis=0)


class CLIPExtractor:
    """Extract patch features from CLIP ViT-L/14 (openai/clip-vit-large-patch14).

    Resolution: 224×224, patch_size=14 → 16×16=256 patches, dim=1024.
    Uses intermediate hidden layer (LAYER_IDX) for richer patch features
    (final layer is optimized for [CLS] token / global representation).
    """
    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, AutoProcessor

        self.device = device
        model_name = "openai/clip-vit-large-patch14"
        print(f"[FEAT] Loading CLIP ViT-L/14 on {device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.patch_size = 14
        self.grid_size = RESOLUTION // self.patch_size  # 224 / 14 = 16
        self.num_patches = self.grid_size * self.grid_size  # 256
        self.feature_dim = self.model.config.vision_config.hidden_size  # 1024

        print(f"[FEAT] Grid: {self.grid_size}x{self.grid_size}, patches: {self.num_patches}, dim: {self.feature_dim}")

    @torch.no_grad()
    def extract_batch(self, image_paths: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(img)

            # Use CLIP processor for consistent preprocessing
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            for img in images:
                img.close()

            with torch.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
                outputs = self.model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )

            # Extract patch tokens from intermediate layer (skip [CLS] token at position 0)
            hidden = outputs.hidden_states[LAYER_IDX]  # (batch, 1+256, 1024)
            patch_features = hidden[:, 1:, :]  # (batch, 256, 1024), drop CLS token

            all_features.append(patch_features.float().cpu().numpy())

            del pixel_values, outputs, hidden, patch_features

        return np.concatenate(all_features, axis=0)


# =============================================================================
# DATASET LOADING (paths + labels only, no features)
# =============================================================================

def load_visa_paths(dataset_path: Path) -> tuple[list[str], list[str], list[int]]:
    """VisA uses CSV for splits."""
    import csv
    category = dataset_path.name
    visa_root = dataset_path.parent
    split_csv = visa_root / "split_csv" / "1cls.csv"

    train_images, test_images, test_labels = [], [], []
    with open(split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['object'] != category:
                continue
            img_path = visa_root / row['image']
            if not img_path.exists():
                continue
            if row['split'] == 'train':
                train_images.append(str(img_path))
            elif row['split'] == 'test':
                label = 0 if row['label'] == 'normal' else 1
                test_images.append(str(img_path))
                test_labels.append(label)
    return sorted(train_images), test_images, test_labels


def load_standard_paths(dataset_path: Path) -> tuple[list[str], list[str], list[int]]:
    """MVTec-style: train/good, test/good+defect."""
    train_dir = dataset_path / "train" / "good"
    if not train_dir.exists():
        train_dir = dataset_path / "train"

    train_images = sorted([str(p) for p in train_dir.rglob("*.png")] +
                         [str(p) for p in train_dir.rglob("*.jpg")] +
                         [str(p) for p in train_dir.rglob("*.JPG")] +
                         [str(p) for p in train_dir.rglob("*.bmp")])

    test_dir = dataset_path / "test"
    test_images, test_labels = [], []
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = 0 if subdir.name == "good" else 1
        for img_path in sorted(list(subdir.rglob("*.png")) +
                               list(subdir.rglob("*.jpg")) +
                               list(subdir.rglob("*.JPG")) +
                               list(subdir.rglob("*.bmp"))):
            test_images.append(str(img_path))
            test_labels.append(label)
    return train_images, test_images, test_labels


def safe_name(dataset_name: str) -> str:
    """Convert 'mvtec_AD/bottle' -> 'mvtec_AD__bottle'"""
    return dataset_name.replace("/", "__")


# =============================================================================
# MAIN
# =============================================================================

def main():
    datasets = ALL_DATASETS[:2] if DEBUG else ALL_DATASETS

    backbone_name = {"dino": "DINOv3-ViT-L", "clip": "CLIP ViT-L/14",
                     "dinov2": "DINOv2-ViT-L/14"}[BACKBONE]
    print("=" * 70)
    print(f"PRECACHE: {backbone_name} Feature Extraction")
    print(f"  Mode: {'DEBUG' if DEBUG else 'FULL'}")
    print(f"  Backbone: {BACKBONE} ({backbone_name})")
    print(f"  Resolution: {RESOLUTION}×{RESOLUTION}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Cache dir: {CACHE_DIR}")
    print("=" * 70)

    if BACKBONE == "dino":
        extractor = DINOv3Extractor(DEVICE)
    elif BACKBONE == "dinov2":
        extractor = DINOv3Extractor(DEVICE, model_name="facebook/dinov2-large",
                                    patch_size=14, num_register=0)
    else:
        extractor = CLIPExtractor(DEVICE)
    t_start = time.time()
    total_images = 0
    skipped = 0

    for ds_idx, dataset_name in enumerate(datasets):
        ds_cache = CACHE_DIR / safe_name(dataset_name)

        # Check if already cached
        if (ds_cache / "test_features.npy").exists() and (ds_cache / "all_train_features.npy").exists():
            print(f"[{ds_idx+1}/{len(datasets)}] {dataset_name}: SKIP (already cached)")
            skipped += 1
            continue

        ds_cache.mkdir(parents=True, exist_ok=True)
        ds_start = time.time()
        print(f"[{ds_idx+1}/{len(datasets)}] {dataset_name}...")

        # Load paths
        dataset_path = DATA_ROOT / dataset_name
        if dataset_name.startswith("VisA/"):
            all_train, test_images, test_labels = load_visa_paths(dataset_path)
        else:
            all_train, test_images, test_labels = load_standard_paths(dataset_path)

        print(f"  train: {len(all_train)} images, test: {len(test_images)} images")

        # Extract ALL train features
        if len(all_train) > 0:
            train_features = extractor.extract_batch(all_train)
            np.save(ds_cache / "all_train_features.npy", train_features)
            with open(ds_cache / "all_train_paths.txt", "w", encoding="utf-8") as f:
                for p in all_train:
                    f.write(p + "\n")
            total_images += len(all_train)
            del train_features

        # Extract ALL test features
        if len(test_images) > 0:
            test_features = extractor.extract_batch(test_images)
            np.save(ds_cache / "test_features.npy", test_features)
            np.save(ds_cache / "test_labels.npy", np.array(test_labels))
            with open(ds_cache / "test_paths.txt", "w", encoding="utf-8") as f:
                for p in test_images:
                    f.write(p + "\n")
            total_images += len(test_images)
            del test_features

        ds_elapsed = time.time() - ds_start
        print(f"  done in {ds_elapsed:.1f}s ({len(all_train) + len(test_images)} images)")

        # Cleanup GPU
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE. {total_images} images extracted in {total_elapsed/60:.1f} min. Skipped: {skipped}")

    # Show cache size
    cache_size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.npy")) / (1024**3)
    print(f"Cache size: {cache_size:.1f} GB")


if __name__ == "__main__":
    main()
