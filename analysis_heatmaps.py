"""
LOO Heatmap Visualizations for Paper 3: Dirty Few-Shot Self-Purification.

Generates qualitative visualizations showing:
1. LOO patch scores overlaid on original images (which patches are flagged)
2. Before vs after purification anomaly score maps
3. Success case (e.g., bottle) and failure case (e.g., LOCO)

Usage:
    python analysis_heatmaps.py                  # Default: bottle + juice_bottle
    python analysis_heatmaps.py --datasets mvtec_AD/carpet VisA/pcb1
    python analysis_heatmaps.py --seed 0 --cont 0.3 --tl 10
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATASETS = ["mvtec_AD/bottle", "mvtec_loco_AD/juice_bottle"]
CACHE_DIR = Path("output/feature_cache")
DEFAULT_KNN_K = 5
DEFAULT_PERCENTILE = 95
RESOLUTION = 448
PATCH_SIZE = 16
GRID_SIZE = 28  # 448 // 16

# ============================================================================
# FEATURE LOADING AND CORE FUNCTIONS
# (Reused from experiment script — minimal self-contained versions)
# ============================================================================

def load_cached_features(dataset_name: str) -> dict:
    """Load pre-cached features for a dataset."""
    safe_name = dataset_name.replace("/", "__")
    cache_path = CACHE_DIR / safe_name

    if not cache_path.exists():
        print(f"  [ERROR] Cache not found: {cache_path}")
        return None

    data = {
        "all_train_features": np.load(str(cache_path / "all_train_features.npy")),
        "test_features": np.load(str(cache_path / "test_features.npy")),
        "test_labels": np.load(str(cache_path / "test_labels.npy")),
    }

    # Load paths
    with open(cache_path / "all_train_paths.txt", "r") as f:
        data["all_train_paths"] = [line.strip() for line in f.readlines()]
    with open(cache_path / "test_paths.txt", "r") as f:
        data["test_paths"] = [line.strip() for line in f.readlines()]

    print(f"  Loaded: {dataset_name} — {data['all_train_features'].shape[0]} train, "
          f"{data['test_features'].shape[0]} test")
    return data


def subsample_and_contaminate(data, train_limit, contamination_rate, seed):
    """Create contaminated training set."""
    rng = np.random.RandomState(seed)
    all_train = data["all_train_features"]
    n_all = all_train.shape[0]

    # Subsample
    if n_all > train_limit:
        indices = sorted(rng.choice(n_all, train_limit, replace=False))
        train_features = all_train[indices].copy()
        train_paths = [data["all_train_paths"][i] for i in indices]
    else:
        train_features = all_train[:train_limit].copy()
        train_paths = data["all_train_paths"][:train_limit]

    # Contaminate
    n_contaminate = int(round(train_limit * contamination_rate))
    is_contaminated = np.zeros(train_limit, dtype=bool)

    if n_contaminate > 0:
        anomaly_indices = np.where(data["test_labels"] == 1)[0]
        if len(anomaly_indices) > 0:
            selected_anomalies = rng.choice(anomaly_indices, n_contaminate, replace=True)
            replace_positions = rng.choice(train_limit, n_contaminate, replace=False)
            for i, pos in enumerate(replace_positions):
                train_features[pos] = data["test_features"][selected_anomalies[i]]
                is_contaminated[pos] = True

    return train_features, train_paths, is_contaminated, n_contaminate


def compute_loo_consistency(features, k=DEFAULT_KNN_K):
    """Compute LOO patch scores and image scores."""
    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False

    n_images, n_patches, dim = features.shape
    patch_scores = np.zeros((n_images, n_patches), dtype=np.float32)

    for i in range(n_images):
        mask = np.ones(n_images, dtype=bool)
        mask[i] = False
        bank = features[mask].reshape(-1, dim).astype(np.float32)
        queries = features[i].astype(np.float32)

        if use_faiss and bank.shape[0] >= k:
            index = faiss.IndexFlatL2(dim)
            index.add(np.ascontiguousarray(bank))
            sq_dists, _ = index.search(np.ascontiguousarray(queries), k)
            distances = np.sqrt(np.maximum(0, sq_dists))
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k, bank.shape[0]), metric="euclidean")
            nn.fit(bank)
            distances, _ = nn.kneighbors(queries)

        patch_scores[i] = distances.mean(axis=1)

    image_scores = patch_scores.mean(axis=1)
    return patch_scores, image_scores


def purify_loo_patch(features, patch_scores, percentile=DEFAULT_PERCENTILE):
    """Remove high-scoring patches. Returns purified bank and removal mask."""
    threshold = np.percentile(patch_scores, percentile)
    remove_mask = patch_scores > threshold  # (n_images, n_patches)
    n_removed = int(remove_mask.sum())

    flat_features = features.reshape(-1, features.shape[-1])
    flat_mask = ~remove_mask.ravel()
    purified_bank = flat_features[flat_mask]

    return purified_bank, n_removed, remove_mask


def compute_anomaly_map(bank, test_features, k=DEFAULT_KNN_K):
    """Compute per-patch anomaly scores for test images."""
    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False

    if bank.ndim == 3:
        bank = bank.reshape(-1, bank.shape[-1])

    bank = bank.astype(np.float32)
    n_test, n_patches, dim = test_features.shape
    patch_scores = np.zeros((n_test, n_patches), dtype=np.float32)

    if use_faiss and bank.shape[0] >= k:
        index = faiss.IndexFlatL2(dim)
        index.add(np.ascontiguousarray(bank))
        queries = test_features.reshape(-1, dim).astype(np.float32)
        sq_dists, _ = index.search(np.ascontiguousarray(queries), k)
        distances = np.sqrt(np.maximum(0, sq_dists))
        patch_scores = distances.mean(axis=1).reshape(n_test, n_patches)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k, bank.shape[0]), metric="euclidean")
        nn.fit(bank)
        queries = test_features.reshape(-1, dim).astype(np.float32)
        distances, _ = nn.kneighbors(queries)
        patch_scores = distances.mean(axis=1).reshape(n_test, n_patches)

    return patch_scores


def patches_to_heatmap(patch_scores, grid_size=GRID_SIZE, target_size=RESOLUTION):
    """Convert flat patch scores to 2D heatmap."""
    grid = patch_scores.reshape(grid_size, grid_size)
    zoom_factor = target_size / grid_size
    heatmap = ndimage_zoom(grid, zoom_factor, order=1)
    return heatmap


def load_image(path: str, size: int = RESOLUTION) -> np.ndarray:
    """Load and resize an image."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


# ============================================================================
# VISUALIZATION: LOO Patch Removal Map
# ============================================================================

def visualize_loo_removal(
    train_paths: list[str],
    is_contaminated: np.ndarray,
    remove_mask: np.ndarray,
    patch_scores: np.ndarray,
    output_path: str,
    dataset_name: str,
    max_images: int = 10,
):
    """
    Show training images with LOO removal overlay.
    Green border = clean, Red border = contaminated.
    Blue patches = removed by LOO.
    """
    n_images = min(len(train_paths), max_images)

    fig, axes = plt.subplots(2, n_images, figsize=(2.5 * n_images, 5.5))
    if n_images == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_images):
        # Top: original image with contamination status
        try:
            img = load_image(train_paths[i])
        except Exception:
            img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)

        axes[0, i].imshow(img)
        border_color = "red" if is_contaminated[i] else "green"
        for spine in axes[0, i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        label = "CONTAM" if is_contaminated[i] else "clean"
        axes[0, i].set_title(f"#{i} ({label})", fontsize=8,
                              color="red" if is_contaminated[i] else "green")

        # Bottom: LOO score heatmap with removal overlay
        heatmap = patches_to_heatmap(patch_scores[i])

        # Create removal overlay
        removal_grid = remove_mask[i].reshape(GRID_SIZE, GRID_SIZE).astype(float)
        removal_up = ndimage_zoom(removal_grid, RESOLUTION / GRID_SIZE, order=0)

        axes[1, i].imshow(img, alpha=0.6)
        axes[1, i].imshow(heatmap, cmap="hot", alpha=0.5,
                          vmin=np.percentile(patch_scores, 10),
                          vmax=np.percentile(patch_scores, 99))

        # Overlay removed patches in blue
        removal_overlay = np.zeros((*removal_up.shape, 4))
        removal_overlay[removal_up > 0.5] = [0, 0.3, 1, 0.4]  # Blue with alpha
        axes[1, i].imshow(removal_overlay)

        n_removed_i = remove_mask[i].sum()
        axes[1, i].set_title(f"LOO (rm={n_removed_i})", fontsize=8)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="green", edgecolor="green", label="Clean image", alpha=0.5),
        mpatches.Patch(facecolor="red", edgecolor="red", label="Contaminated image", alpha=0.5),
        mpatches.Patch(facecolor="blue", edgecolor="blue", label="Removed patches", alpha=0.4),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8)

    fig.suptitle(f"LOO Patch Removal — {dataset_name}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# VISUALIZATION: Before/After Purification Anomaly Maps
# ============================================================================

def _load_gt_contour(test_path: str, dataset_name: str, size: int = RESOLUTION):
    """Load GT mask and return a binary contour array for overlay."""
    from pathlib import Path as _Path
    from PIL import Image as _Img

    p = _Path(test_path)
    parts = p.parts

    # --- Resolve mask path (inline, avoids importing the experiment script) ---
    mask = None

    if "VisA" in dataset_name:
        if "Normal" not in test_path:
            mask_path = _Path(test_path.replace("Images", "Masks")).with_suffix(".png")
            if mask_path.exists():
                mask = np.array(_Img.open(mask_path).convert("L"))
    else:
        # MVTec AD / BTAD / LOCO
        category = parts[-2]
        if category != "good":
            test_idx = None
            for i, part in enumerate(parts):
                if part == "test":
                    test_idx = i
                    break
            if test_idx is not None:
                data_root = _Path(*parts[:test_idx])
                gt_dir = data_root / "ground_truth"
                stem = p.stem

                if "mvtec_loco_AD" in dataset_name:
                    mask_subdir = gt_dir / category / stem
                    if mask_subdir.exists():
                        combined = None
                        for mp in mask_subdir.glob("*.png"):
                            m = np.array(_Img.open(mp).convert("L"))
                            combined = m if combined is None else np.maximum(combined, m)
                        mask = combined
                else:
                    mask_dir = gt_dir / category
                    if mask_dir.exists():
                        for ext in [".png", "_mask.png"]:
                            mp = mask_dir / f"{stem}{ext}"
                            if mp.exists():
                                mask = np.array(_Img.open(mp).convert("L"))
                                break

    if mask is None:
        return None

    # Resize mask to target size and extract contour
    mask_resized = np.array(_Img.fromarray(mask).resize((size, size), _Img.NEAREST))
    binary = (mask_resized > 127).astype(np.uint8)

    # Extract contour using morphological gradient
    from scipy.ndimage import binary_dilation, binary_erosion
    dilated = binary_dilation(binary, iterations=4)
    eroded = binary_erosion(binary, iterations=1)
    contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    contour = np.clip(contour, 0, 1)
    return contour


def visualize_before_after(
    test_paths: list[str],
    test_labels: np.ndarray,
    maps_before: np.ndarray,
    maps_after: np.ndarray,
    output_path: str,
    dataset_name: str,
    n_show: int = 8,
):
    """
    Show test images with anomaly heatmaps before and after purification.
    Select a mix of normal and anomalous images.
    """
    anomaly_idx = np.where(test_labels == 1)[0]
    normal_idx = np.where(test_labels == 0)[0]

    # Select images to show
    n_anom = min(n_show // 2, len(anomaly_idx))
    n_norm = min(n_show - n_anom, len(normal_idx))

    # Pick anomalous images with highest score difference
    score_diff = maps_after.mean(axis=1) - maps_before.mean(axis=1)
    if len(anomaly_idx) > n_anom:
        sorted_anom = anomaly_idx[np.argsort(-np.abs(score_diff[anomaly_idx]))]
        selected_anom = sorted_anom[:n_anom]
    else:
        selected_anom = anomaly_idx[:n_anom]

    if len(normal_idx) > n_norm:
        rng = np.random.RandomState(42)
        selected_norm = rng.choice(normal_idx, n_norm, replace=False)
    else:
        selected_norm = normal_idx[:n_norm]

    selected = np.concatenate([selected_anom, selected_norm])
    n_total = len(selected)

    if n_total == 0:
        print("  [SKIP] No images to show")
        return

    fig, axes = plt.subplots(3, n_total, figsize=(2.5 * n_total, 7.5))
    if n_total == 1:
        axes = axes.reshape(3, 1)

    vmin = min(np.percentile(maps_before, 5), np.percentile(maps_after, 5))
    vmax = max(np.percentile(maps_before, 98), np.percentile(maps_after, 98))

    for col, idx in enumerate(selected):
        try:
            img = load_image(test_paths[idx])
        except Exception:
            img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)

        is_anomaly = test_labels[idx] == 1
        label = "Anomaly" if is_anomaly else "Normal"
        color = "red" if is_anomaly else "green"

        # Load GT contour for anomaly images
        gt_contour = None
        if is_anomaly:
            try:
                gt_contour = _load_gt_contour(test_paths[idx], dataset_name)
            except Exception:
                pass

        # Row 0: Original image + GT contour
        axes[0, col].imshow(img)
        if gt_contour is not None:
            contour_overlay = np.zeros((*gt_contour.shape, 4))
            contour_overlay[gt_contour > 0] = [1, 0, 0, 0.9]
            axes[0, col].imshow(contour_overlay)
        axes[0, col].set_title(label, fontsize=9, color=color, fontweight="bold")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Row 1: Before purification
        hm_before = patches_to_heatmap(maps_before[idx])
        axes[1, col].imshow(img, alpha=0.3)
        axes[1, col].imshow(hm_before, cmap="hot", alpha=0.7, vmin=vmin, vmax=vmax)
        score_before = np.percentile(maps_before[idx], 95)
        axes[1, col].set_title(f"Dirty ({score_before:.1f})", fontsize=8)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # Row 2: After purification
        hm_after = patches_to_heatmap(maps_after[idx])
        axes[2, col].imshow(img, alpha=0.3)
        axes[2, col].imshow(hm_after, cmap="hot", alpha=0.7, vmin=vmin, vmax=vmax)
        score_after = np.percentile(maps_after[idx], 95)
        axes[2, col].set_title(f"Purified ({score_after:.1f})", fontsize=8)
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Dirty bank", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Purified bank", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# COMBINED MULTI-CLASS VISUALIZATION
# ============================================================================

COMBINED_STRUCTURAL = [
    {"name": "btad/03", "anomaly_rank": 0, "normal_rank": 0, "display": "BTAD 03"},
    {"name": "mvtec_AD/bottle", "anomaly_rank": 1, "normal_rank": 0, "display": "Bottle"},
    {"name": "mvtec_AD/tile", "anomaly_rank": 0, "normal_rank": 0, "display": "Tile"},
    {"name": "mvtec_AD/toothbrush", "anomaly_rank": 0, "normal_rank": 0, "display": "Toothbrush"},
]

COMBINED_LOCO = [
    {"name": "mvtec_loco_AD/juice_bottle", "anomaly_rank": 0, "normal_rank": 0,
     "display": "Juice Bottle", "select": "median"},
    {"name": "mvtec_loco_AD/breakfast_box", "anomaly_rank": 0, "normal_rank": 0,
     "display": "Breakfast Box", "select": "median"},
    {"name": "mvtec_loco_AD/pushpins", "anomaly_rank": 0, "normal_rank": 0,
     "display": "Pushpins", "select": "median"},
    {"name": "mvtec_loco_AD/screw_bag", "anomaly_rank": 0, "normal_rank": 0,
     "display": "Screw Bag", "select": "median"},
]


def process_for_combined(dataset_name, args):
    """Process a dataset and return data needed for combined figure."""
    data = load_cached_features(dataset_name)
    if data is None:
        return None

    train_features, train_paths, is_contaminated, n_contam = subsample_and_contaminate(
        data, args.tl, args.cont, args.seed,
    )

    print(f"  Computing LOO for {dataset_name}...")
    patch_scores, image_scores = compute_loo_consistency(train_features, k=args.knn_k)

    purified_bank, n_removed, remove_mask = purify_loo_patch(
        train_features, patch_scores, percentile=args.percentile,
    )

    print(f"  Computing anomaly maps for {dataset_name}...")
    maps_before = compute_anomaly_map(train_features, data["test_features"], k=args.knn_k)
    maps_after = compute_anomaly_map(purified_bank, data["test_features"], k=args.knn_k)

    # Report contamination stats
    if n_contam > 0:
        contam_removed = remove_mask[is_contaminated].sum()
        contam_total = is_contaminated.sum() * patch_scores.shape[1]
        clean_removed = remove_mask[~is_contaminated].sum()
        clean_total = (~is_contaminated).sum() * patch_scores.shape[1]
        print(f"    Contam patches removed: {contam_removed}/{contam_total} "
              f"({100*contam_removed/max(contam_total,1):.1f}%)")
        print(f"    Clean patches removed: {clean_removed}/{clean_total} "
              f"({100*clean_removed/max(clean_total,1):.1f}%)")

    return {
        "test_paths": data["test_paths"],
        "test_labels": data["test_labels"],
        "maps_before": maps_before,
        "maps_after": maps_after,
    }


def select_images(result, anomaly_rank, normal_rank, select_method="max_diff"):
    """Select specific anomaly and normal test images by rank.

    select_method:
        "max_diff" — sort anomalies by largest |score change| (default, for structural).
        "median"   — pick the anomaly closest to the median change (for LOCO).
    """
    labels = result["test_labels"]
    maps_before = result["maps_before"]
    maps_after = result["maps_after"]

    anomaly_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]

    score_diff = maps_after.mean(axis=1) - maps_before.mean(axis=1)

    if select_method == "median":
        # Sort by smallest absolute difference → most representative / least changed
        sorted_anom = anomaly_idx[np.argsort(np.abs(score_diff[anomaly_idx]))]
        # Pick from the middle of this sorted list
        mid = len(sorted_anom) // 2
        a_idx = int(sorted_anom[mid + anomaly_rank])
    else:
        # Sort by largest absolute difference → most dramatic change
        sorted_anom = anomaly_idx[np.argsort(-np.abs(score_diff[anomaly_idx]))]
        a_rank = min(anomaly_rank, len(sorted_anom) - 1)
        a_idx = int(sorted_anom[a_rank])

    # Deterministic normal selection (same seed as visualize_before_after)
    rng = np.random.RandomState(42)
    n_norm = min(len(normal_idx), 4)
    shuffled_norm = rng.choice(normal_idx, n_norm, replace=False)
    n_rank = min(normal_rank, len(shuffled_norm) - 1)

    return a_idx, int(shuffled_norm[n_rank])


def visualize_combined(configs, results, output_path, is_loco=False):
    """
    Create combined multi-class before/after figure with global normalization.

    Landscape layout: 3 rows × (2 × n_classes) columns.
    Left half: anomaly samples.  Right half: normal samples.
    Rows: test image | contaminated bank | purified bank.
    """
    from matplotlib.gridspec import GridSpec

    n_classes = len(configs)
    n_cols = 2 * n_classes  # anomaly cols + normal cols

    # Select images for each class
    selected = []
    for cfg, res in zip(configs, results):
        method = cfg.get("select", "max_diff")
        anom_idx, norm_idx = select_images(
            res, cfg["anomaly_rank"], cfg["normal_rank"], select_method=method,
        )
        selected.append((anom_idx, norm_idx))
        score_b = np.percentile(res["maps_before"][anom_idx], 95)
        score_a = np.percentile(res["maps_after"][anom_idx], 95)
        print(f"  {cfg['display']}: anomaly #{anom_idx} "
              f"(dirty={score_b:.0f} → purified={score_a:.0f})")

    # ---- Global normalization across ALL selected images ----
    all_values = []
    for (anom_idx, norm_idx), res in zip(selected, results):
        for idx in [anom_idx, norm_idx]:
            all_values.append(res["maps_before"][idx])
            all_values.append(res["maps_after"][idx])
    all_values = np.concatenate(all_values)

    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)
    print(f"  Global normalization: vmin={vmin:.1f}, vmax={vmax:.1f}")

    # ---- Create figure ----
    # Wide landscape: 8 cols × 3 rows, tight spacing
    fig_w = 1.55 * n_cols + 1.0
    fig_h = 4.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Width ratios: equal image cols with a small gap between the two halves
    width_ratios = [1] * n_classes + [0.06] + [1] * n_classes
    total_cols = n_cols + 1  # +1 for separator column

    gs = GridSpec(
        3, total_cols,
        figure=fig,
        hspace=0.04,
        wspace=0.03,
        width_ratios=width_ratios,
        left=0.06, right=0.99, top=0.90, bottom=0.01,
    )

    row_labels = ["Test image", "Contam. bank", "Purified bank"]
    sep_col = n_classes  # separator column index

    # Column mapping: anomaly cols 0..n_classes-1, normal cols n_classes+1..2*n_classes
    def col_index(block, class_idx):
        if block == "anomaly":
            return class_idx
        return n_classes + 1 + class_idx  # +1 skips separator

    for ci, (cfg, res, (anom_idx, norm_idx)) in enumerate(
        zip(configs, results, selected)
    ):
        for block_name, test_idx in [
            ("anomaly", anom_idx),
            ("normal", norm_idx),
        ]:
            is_anomaly = block_name == "anomaly"
            gcol = col_index(block_name, ci)

            # Load test image
            try:
                img = load_image(res["test_paths"][test_idx])
            except Exception:
                img = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)

            # --- Row 0: Original image ---
            ax0 = fig.add_subplot(gs[0, gcol])
            ax0.imshow(img)
            if is_anomaly:
                try:
                    gt = _load_gt_contour(res["test_paths"][test_idx], cfg["name"])
                    if gt is not None:
                        overlay = np.zeros((*gt.shape, 4))
                        overlay[gt > 0] = [1, 0, 0, 0.9]
                        ax0.imshow(overlay)
                except Exception:
                    pass
            ax0.set_xticks([])
            ax0.set_yticks([])
            # Column header
            ax0.set_title(cfg["display"], fontsize=10, fontweight="bold", pad=3)
            # Row label on leftmost column only
            if gcol == 0:
                ax0.set_ylabel(row_labels[0], fontsize=10, fontweight="bold")

            # --- Row 1: Contaminated bank heatmap ---
            ax1 = fig.add_subplot(gs[1, gcol])
            hm_b = patches_to_heatmap(res["maps_before"][test_idx])
            ax1.imshow(img, alpha=0.3)
            ax1.imshow(hm_b, cmap="hot", alpha=0.7, vmin=vmin, vmax=vmax)
            ax1.set_xticks([])
            ax1.set_yticks([])
            if gcol == 0:
                ax1.set_ylabel(row_labels[1], fontsize=10, fontweight="bold")

            # --- Row 2: Purified bank heatmap ---
            ax2 = fig.add_subplot(gs[2, gcol])
            hm_a = patches_to_heatmap(res["maps_after"][test_idx])
            ax2.imshow(img, alpha=0.3)
            ax2.imshow(hm_a, cmap="hot", alpha=0.7, vmin=vmin, vmax=vmax)
            ax2.set_xticks([])
            ax2.set_yticks([])
            if gcol == 0:
                ax2.set_ylabel(row_labels[2], fontsize=10, fontweight="bold")

    # Hide separator column
    for r in range(3):
        ax_sep = fig.add_subplot(gs[r, sep_col])
        ax_sep.set_visible(False)

    # Block group labels above the column headers
    # Anomaly block spans columns 0..n_classes-1
    anom_left = gs[0, 0].get_position(fig).x0
    anom_right = gs[0, n_classes - 1].get_position(fig).x1
    fig.text(
        (anom_left + anom_right) / 2, 0.97,
        "Anomaly samples", fontsize=12, fontweight="bold",
        ha="center", va="center", color="red",
    )
    # Normal block spans columns n_classes+1..total_cols-1
    norm_left = gs[0, n_classes + 1].get_position(fig).x0
    norm_right = gs[0, total_cols - 1].get_position(fig).x1
    fig.text(
        (norm_left + norm_right) / 2, 0.97,
        "Normal samples", fontsize=12, fontweight="bold",
        ha="center", va="center", color="green",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_combined_figures(args, output_dir):
    """Generate both combined structural and LOCO figures."""
    # --- Structural figure ---
    print(f"\n{'=' * 60}")
    print("  COMBINED STRUCTURAL FIGURE")
    print(f"{'=' * 60}")

    structural_results = []
    for cfg in COMBINED_STRUCTURAL:
        res = process_for_combined(cfg["name"], args)
        if res is None:
            print(f"  [SKIP] {cfg['name']}")
            return
        structural_results.append(res)

    visualize_combined(
        COMBINED_STRUCTURAL, structural_results,
        str(output_dir / "heatmap_combined_structural.pdf"),
        is_loco=False,
    )

    # --- LOCO figure ---
    print(f"\n{'=' * 60}")
    print("  COMBINED LOCO FIGURE")
    print(f"{'=' * 60}")

    loco_results = []
    for cfg in COMBINED_LOCO:
        res = process_for_combined(cfg["name"], args)
        if res is None:
            print(f"  [SKIP] {cfg['name']}")
            return
        loco_results.append(res)

    visualize_combined(
        COMBINED_LOCO, loco_results,
        str(output_dir / "heatmap_combined_loco.pdf"),
        is_loco=True,
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_dataset(dataset_name, args, output_dir):
    """Run full heatmap visualization pipeline for one dataset."""
    print(f"\n{'=' * 60}")
    print(f"  Processing: {dataset_name}")
    print(f"{'=' * 60}")

    # Load features
    data = load_cached_features(dataset_name)
    if data is None:
        return

    # Create contaminated train set
    train_features, train_paths, is_contaminated, n_contam = subsample_and_contaminate(
        data, args.tl, args.cont, args.seed,
    )
    print(f"  Train: {train_features.shape[0]} images, {n_contam} contaminated")

    # Compute LOO consistency
    print("  Computing LOO consistency...")
    patch_scores, image_scores = compute_loo_consistency(train_features, k=args.knn_k)
    print(f"  LOO scores: patch shape {patch_scores.shape}, image shape {image_scores.shape}")

    # Purify
    print("  Purifying...")
    purified_bank, n_removed, remove_mask = purify_loo_patch(
        train_features, patch_scores, percentile=args.percentile,
    )
    print(f"  Removed {n_removed} patches ({n_removed / patch_scores.size * 100:.1f}%)")

    # Check: how many contaminated patches were caught?
    if n_contam > 0:
        contam_removed = remove_mask[is_contaminated].sum()
        clean_removed = remove_mask[~is_contaminated].sum()
        contam_total = is_contaminated.sum() * patch_scores.shape[1]
        clean_total = (~is_contaminated).sum() * patch_scores.shape[1]
        print(f"  Patches removed from contaminated images: {contam_removed}/{contam_total} "
              f"({100*contam_removed/max(contam_total,1):.1f}%)")
        print(f"  Patches removed from clean images: {clean_removed}/{clean_total} "
              f"({100*clean_removed/max(clean_total,1):.1f}%)")

    # Compute anomaly maps before/after
    n_test_show = min(len(data["test_features"]), 200)  # Limit for memory
    test_features = data["test_features"][:n_test_show]
    test_labels = data["test_labels"][:n_test_show]
    test_paths = data["test_paths"][:n_test_show]

    print("  Computing anomaly maps (before purification)...")
    dirty_bank = train_features  # 3D
    maps_before = compute_anomaly_map(dirty_bank, test_features, k=args.knn_k)

    print("  Computing anomaly maps (after purification)...")
    maps_after = compute_anomaly_map(purified_bank, test_features, k=args.knn_k)

    # Generate visualizations
    safe_name = dataset_name.replace("/", "_")

    # 1. LOO removal visualization
    print("  Generating LOO removal visualization...")
    visualize_loo_removal(
        train_paths, is_contaminated, remove_mask, patch_scores,
        str(output_dir / f"heatmap_loo_removal_{safe_name}.pdf"),
        dataset_name, max_images=min(10, args.tl),
    )

    # 2. Before/after anomaly maps
    print("  Generating before/after anomaly maps...")
    visualize_before_after(
        test_paths, test_labels, maps_before, maps_after,
        str(output_dir / f"heatmap_before_after_{safe_name}.pdf"),
        dataset_name, n_show=8,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate LOO heatmap visualizations")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Datasets to visualize")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tl", type=int, default=10, help="Train limit")
    parser.add_argument("--cont", type=float, default=0.3, help="Contamination rate")
    parser.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    parser.add_argument("--percentile", type=int, default=DEFAULT_PERCENTILE)
    parser.add_argument("--combined", action="store_true",
                        help="Generate combined multi-class figures only")
    args = parser.parse_args()

    output_dir = Path("output/analysis/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  LOO HEATMAP VISUALIZATIONS")
    print(f"  Seed: {args.seed}, TL: {args.tl}, Cont: {args.cont}")
    print(f"{'#' * 70}")

    if args.combined:
        generate_combined_figures(args, output_dir)
    else:
        for dataset in args.datasets:
            try:
                process_dataset(dataset, args, output_dir)
            except Exception as e:
                print(f"  [ERROR] {dataset}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — heatmaps saved to {output_dir}")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
