"""
P4: Image-level purification under shared (A) and disjoint (B) contamination protocols.

Motivation (from the 35-category disjoint-pool analysis):
  - Real contamination damage (Setup B) is ~1.7 AUROC pts at c=0.3 (significant).
  - Patch-level purification does not recover it (p~0.5), but the image-level
    oracle recovers 58-72%. This experiment tests PRACTICAL image-level rules.

Methods evaluated per cell (sharing one LOO computation). Each image gets a
score by aggregating its per-patch leave-one-out distances, and images whose
score exceeds a robust threshold (median + k * 1.4826 * MAD, capped at 40% of N)
are removed:
  - none          : no screening (baseline; also validates protocol fidelity
                    against results_v2.csv / results_leakage_check.csv).
  - imgmax_mad    : score = MAX per-patch LOO distance, k in {1.0, 1.5, 2.0}.
                    This is the paper's deployable rule (works at c=0 too).
  - imgq99_mad,
    imgt5_mad     : robustness ablation, score = 99th percentile / top-5% mean.
  - image_auto_mad: score = MEAN per-patch distance, k in {2.0, 2.5, 3.0}
                    (the mean-pooled variant that the paper shows fails).
  - *_known_c     : remove exactly ceil(N*c) top-scoring images (upper bound
                    that assumes the contamination rate is known).

Setups:
  A (shared)  : contaminants drawn from full test anomaly pool, evaluate on full
                test set (mirrors the main protocol in results_v2.csv).
  B (disjoint): anomalies split 50/50 with RandomState(seed+5000) shuffle, pool A
                contaminates, pool B + all normals evaluate (mirrors
                run_leakage_experiment / results_leakage_check.csv EXACTLY).

Self-contained, resumable, throttled. Does NOT modify the main experiment script.

Usage:
  python exp_p4_image_level.py --smoke                 # ~2 min validation run
  python exp_p4_image_level.py --setup B               # full disjoint run (priority)
  python exp_p4_image_level.py --setup A               # full shared run (tables)
  python exp_p4_image_level.py --setup both
Options: --threads N (default 12) --sleep S (default 5) --max-datasets N
"""
import os
import sys
import csv
import time
import argparse
from pathlib import Path


def set_thread_env(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def set_low_priority() -> str:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        return "below-normal (psutil)"
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        ok = k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        return "below-normal (ctypes)" if ok else "default"
    except Exception as e:
        return f"default ({e})"


FIELDNAMES = [
    "timestamp", "dataset", "setup", "config", "method", "param_k",
    "contamination_rate", "seed", "train_limit", "knn_k",
    "n_train_original", "n_contaminated",
    "n_images_removed", "n_contaminated_removed",
    "removal_precision", "removal_recall", "image_score_det_auroc",
    "n_patches_after",
    "n_test", "n_normal_test", "n_anomaly_test",
    "img_auroc", "img_aupr", "img_f1max", "tpr_fpr1", "tpr_fpr5",
    "elapsed_s",
]

AUTO_KS = [2.0, 2.5, 3.0]      # MAD multipliers for the mean-LOO image score
MAX_KS = [1.0, 1.5, 2.0]        # MAD multipliers for the max-LOO image score
# Robust-aggregation ablation (Setup B only): does the result depend on a single
# extreme patch? q99 and top-5%-mean are robust alternatives to the plain max.
AGG_EXTRA = [("imgq99_mad", "1.5"), ("imgt5_mad", "1.5")]


def cell_methods(setup: str, c: float) -> list:
    """Methods evaluated for one (setup, contamination) cell."""
    m = [("none", "")]
    if c > 0:
        m += [("image_known_c", ""), ("imgmax_known_c", "")]
    m += [("image_auto_mad", k) for k in AUTO_KS]
    m += [("imgmax_mad", k) for k in MAX_KS]
    if setup == "B":
        m += AGG_EXTRA
    return m


def row_key(r: dict) -> tuple:
    return (r["dataset"], r["setup"], r["method"], str(r["param_k"]),
            str(r["contamination_rate"]), str(r["seed"]), str(r["train_limit"]))


def get_completed(csv_path: Path) -> set:
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                done.add(row_key(r))
    return done


def save_row(row: dict, csv_path: Path) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["A", "B", "both"], default="B")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--sleep", type=int, default=5)
    ap.add_argument("--max-datasets", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tl", default="", help="comma list of train limits, e.g. '5' (default: 10,20)")
    ap.add_argument("--conts", default="", help="comma list of contamination rates (default: 0,0.1,0.2,0.3)")
    ap.add_argument("--backbone", choices=["dino", "clip", "dinov2"], default="dino")
    args = ap.parse_args()

    set_thread_env(args.threads)
    prio = set_low_priority()

    import numpy as np
    import exp_p3_002_dirty_fewshot_full as E
    try:
        import faiss
        faiss.omp_set_num_threads(args.threads)
    except Exception:
        pass
    from sklearn.metrics import roc_auc_score

    out_dir = Path("output/exp_p4_image_level")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.backbone == "clip":
        E.CACHE_DIR = Path("output/feature_cache_clip")
        csv_path = out_dir / "results_clip.csv"
    elif args.backbone == "dinov2":
        E.CACHE_DIR = Path("output/feature_cache_dinov2")
        csv_path = out_dir / "results_dinov2.csv"
    else:
        csv_path = out_dir / "results.csv"
    status_path = Path("exp_p4_status.txt")

    def status(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        try:
            with open(status_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    knn_k = E.DEFAULT_KNN_K
    setups = ["B", "A"] if args.setup == "both" else [args.setup]

    if args.smoke:
        datasets = ["mvtec_AD/bottle"]
        seeds, train_limits, cont_rates = [0], [10], [0.0, 0.3]
        setups = ["B", "A"]
    else:
        datasets = list(E.ALL_DATASETS)
        seeds, train_limits, cont_rates = E.SEEDS, E.TRAIN_LIMITS, E.CONTAMINATION_RATES
        if args.tl:
            train_limits = [int(x) for x in args.tl.split(",")]
        if args.conts:
            cont_rates = [float(x) for x in args.conts.split(",")]

    completed = get_completed(csv_path)
    status("=" * 60)
    status(f"START p4 image-level | backbone={args.backbone} setups={setups} tl={train_limits} "
           f"conts={cont_rates} threads={args.threads} priority={prio} smoke={args.smoke} "
           f"| done_rows={len(completed)}")

    def purify_auto_mad(image_scores: np.ndarray, k_sigma: float) -> np.ndarray:
        """Return boolean remove-mask over images using median + k*1.4826*MAD."""
        med = float(np.median(image_scores))
        mad = float(np.median(np.abs(image_scores - med)))
        n = len(image_scores)
        if mad < 1e-12:
            return np.zeros(n, dtype=bool)
        thr = med + k_sigma * 1.4826 * mad
        mask = image_scores > thr
        cap = int(0.4 * n)
        if mask.sum() > cap:
            # keep only the top-cap offenders
            order = np.argsort(image_scores)[::-1]
            mask = np.zeros(n, dtype=bool)
            mask[order[:cap]] = True
        return mask

    def eval_bank(bank, eval_feats, cache_key, none_cache):
        """kNN score + metrics, with reuse when the bank is the unpurified one."""
        scores = E.compute_knn_scores(bank, eval_feats, k=knn_k)
        return scores

    processed = 0
    for ds_idx, ds in enumerate(datasets):
        todo = False
        for setup in setups:
            for seed in seeds:
                for tl in train_limits:
                    for c in cont_rates:
                        for meth, par in cell_methods(setup, c):
                            if (ds, setup, meth, str(par), str(c), str(seed), str(tl)) not in completed:
                                todo = True
        if not todo:
            status(f"[{ds_idx+1}/{len(datasets)}] {ds}: complete, skipping")
            continue

        try:
            cached = E.load_cached_dataset(ds)
        except FileNotFoundError as e:
            status(f"[{ds_idx+1}/{len(datasets)}] {ds}: CACHE MISSING ({e}), skipping")
            continue

        status(f"[{ds_idx+1}/{len(datasets)}] {ds}: running")
        t_ds = time.time()

        for setup in setups:
            for seed in seeds:
                for tl in train_limits:
                    train_features, anomaly_indices, test_features, test_labels, _ = \
                        E.prepare_experiment_data(cached, tl, seed)

                    if setup == "B":
                        # EXACT mirror of run_leakage_experiment split
                        rng_split = np.random.RandomState(seed + 5000)
                        arr = np.array(anomaly_indices)
                        rng_split.shuffle(arr)
                        n_pool_a = len(arr) // 2
                        pool_a = arr[:n_pool_a]
                        pool_b = arr[n_pool_a:]
                        normal_idx = np.where(test_labels == 0)[0]
                        eval_idx = np.sort(np.concatenate([normal_idx, pool_b]))
                        eval_feats = test_features[eval_idx]
                        eval_labels = test_labels[eval_idx]
                        contam_pool = test_features[pool_a]
                    else:
                        eval_feats = test_features
                        eval_labels = test_labels
                        contam_pool = test_features[np.array(anomaly_indices)]

                    if contam_pool.shape[0] == 0 or (eval_labels == 1).sum() == 0:
                        status(f"    [WARN] {ds} setup={setup} seed={seed} tl={tl}: empty pool, skip")
                        continue

                    n_test = eval_feats.shape[0]
                    n_norm = int((eval_labels == 0).sum())
                    n_anom = int((eval_labels == 1).sum())

                    for c in cont_rates:
                        methods = cell_methods(setup, c)
                        keys = {(meth, str(par)): (ds, setup, meth, str(par), str(c), str(seed), str(tl))
                                for meth, par in methods}
                        if all(k in completed for k in keys.values()):
                            continue

                        t0 = time.time()
                        if c == 0.0:
                            bank3d = train_features.copy()
                            is_cont = [0] * train_features.shape[0]
                        else:
                            bank3d, is_cont = E.contaminate_train_features(
                                train_features, contam_pool, c, seed)
                        n_images = bank3d.shape[0]
                        n_cont = int(sum(is_cont))

                        # one LOO computation shared by all methods; image score =
                        # aggregation of the per-patch scores. Defects are localized,
                        # so upper-tail aggregations (max, q99, top-5% mean) carry the
                        # signal while the plain mean dilutes it.
                        patch_scores, image_scores_mean = E.compute_loo_consistency(bank3d, k=knn_k)
                        n_top = max(1, int(np.ceil(0.05 * patch_scores.shape[1])))
                        agg_scores = {
                            "image": image_scores_mean,
                            "imgmax": patch_scores.max(axis=1),
                            "imgq99": np.percentile(patch_scores, 99, axis=1),
                            "imgt5": np.sort(patch_scores, axis=1)[:, -n_top:].mean(axis=1),
                        }
                        det_by_fam = {}
                        if 0 < n_cont < n_images:
                            for fam, sc in agg_scores.items():
                                det_by_fam[fam] = float(roc_auc_score(is_cont, sc))

                        none_scores = None
                        none_metrics = None

                        for meth, par in methods:
                            key = keys[(meth, str(par))]
                            if key in completed:
                                continue
                            t1 = time.time()

                            fam = "image" if meth in ("none", "image_known_c", "image_auto_mad") \
                                else meth.split("_")[0]
                            image_scores = agg_scores[fam]
                            det_auroc = det_by_fam.get(fam, "")

                            if meth == "none":
                                remove_mask = np.zeros(n_images, dtype=bool)
                            elif meth in ("image_known_c", "imgmax_known_c"):
                                n_rm = int(np.ceil(n_images * c))
                                n_rm = min(n_rm, n_images - 1)
                                order = np.argsort(image_scores)[::-1]
                                remove_mask = np.zeros(n_images, dtype=bool)
                                remove_mask[order[:n_rm]] = True
                            else:  # *_mad rules
                                remove_mask = purify_auto_mad(image_scores, float(par))

                            n_rm = int(remove_mask.sum())
                            n_cont_rm = int(sum(1 for i in range(n_images)
                                                if remove_mask[i] and is_cont[i]))
                            prec = (n_cont_rm / n_rm) if n_rm > 0 else ""
                            rec = (n_cont_rm / n_cont) if n_cont > 0 else ""

                            if n_rm == 0:
                                if none_scores is None:
                                    bank = bank3d.reshape(-1, bank3d.shape[-1])
                                    none_scores = E.compute_knn_scores(bank, eval_feats, k=knn_k)
                                    none_metrics = E.compute_metrics(eval_labels, none_scores)
                                metrics = none_metrics
                                n_patches_after = n_images * bank3d.shape[1]
                            else:
                                kept = bank3d[~remove_mask]
                                bank = kept.reshape(-1, kept.shape[-1])
                                if bank.shape[0] < knn_k:
                                    status(f"    [WARN] bank too small after removal, skip {key}")
                                    continue
                                scores = E.compute_knn_scores(bank, eval_feats, k=knn_k)
                                metrics = E.compute_metrics(eval_labels, scores)
                                n_patches_after = bank.shape[0]

                            row = {
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "dataset": ds, "setup": setup,
                                "config": f"{setup}_c{int(c*100)}_{meth}{par}",
                                "method": meth, "param_k": par,
                                "contamination_rate": c, "seed": seed,
                                "train_limit": tl, "knn_k": knn_k,
                                "n_train_original": n_images, "n_contaminated": n_cont,
                                "n_images_removed": n_rm,
                                "n_contaminated_removed": n_cont_rm,
                                "removal_precision": prec, "removal_recall": rec,
                                "image_score_det_auroc": det_auroc,
                                "n_patches_after": n_patches_after,
                                "n_test": n_test, "n_normal_test": n_norm,
                                "n_anomaly_test": n_anom,
                                "img_auroc": round(metrics["img_auroc"], 6),
                                "img_aupr": round(metrics["img_aupr"], 6),
                                "img_f1max": round(metrics["img_f1max"], 6),
                                "tpr_fpr1": round(metrics["tpr_fpr1"], 6),
                                "tpr_fpr5": round(metrics["tpr_fpr5"], 6),
                                "elapsed_s": round(time.time() - t1, 2),
                            }
                            save_row(row, csv_path)
                            completed.add(key)
                            print(f"    {row['config']} seed={seed} tl={tl}: "
                                  f"AUROC={row['img_auroc']:.4f} rm={n_rm}/{n_images} "
                                  f"(cont_rm={n_cont_rm}/{n_cont})", flush=True)

        del cached
        import gc
        gc.collect()
        status(f"[{ds_idx+1}/{len(datasets)}] {ds}: done in {(time.time()-t_ds)/60:.1f} min")
        processed += 1
        if args.max_datasets and processed >= args.max_datasets:
            status(f"Reached --max-datasets={args.max_datasets}, stopping (resumable).")
            break
        if not args.smoke:
            time.sleep(args.sleep)

    status(f"END p4 image-level | datasets processed: {processed}")


if __name__ == "__main__":
    main()
