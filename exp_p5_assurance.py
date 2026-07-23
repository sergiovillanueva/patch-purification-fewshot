"""
P5: Commissioning-time reference assurance experiments (decoupled auditor).

Three questions, one shared computation (Setup B, N=10, disjoint protocol):
  A. AUDITOR TRANSFER: DINOv3 max-LOO screening decisions applied to the banks
     of other detectors (CLIP, DINOv2). Converts the self-audit null results
     into a design motivation if the transfer works. Oracle removal per
     detector gives each detector's ceiling.
  C. REVIEW BUDGET: fraction of contaminated references found in the top-1/2/3
     of the auditor ranking (operational reading of recall).
  D. RECAPTURE POLICY: flagged images are replaced by unused normal images
     from the training pool (constant bank size), evaluated on the DINOv3
     detector at c=0.3 and c=0 (clean cost).
  E. AUDIT BASELINES: LOF / IsolationForest / kNN distance on global
     (mean-pooled) image embeddings, same MAD threshold, DINOv3 detector.

Everything is index-level where possible: contamination and removal decisions
are image indices, so one auditor decision applies to any detector's cached
features. Per dataset the caches are loaded one backbone at a time to bound
memory. Resumable CSV. Does not modify existing scripts.

Usage:
  python exp_p5_assurance.py --smoke          # bottle, seed 0, ~3 min
  python exp_p5_assurance.py                  # full run
Options: --threads N (default 12) --sleep S (default 5) --max-datasets N
"""
import os
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
        return "below-normal"
    except Exception:
        return "default"


FIELDNAMES = [
    "timestamp", "dataset", "setup", "detector", "auditor", "method",
    "contamination_rate", "seed", "train_limit",
    "n_train", "n_contaminated", "n_removed", "n_cont_removed",
    "top1_cont", "top2_cont", "top3_cont",
    "img_auroc", "img_aupr", "tpr_fpr1", "tpr_fpr5",
    "elapsed_s",
]

CACHES = {"dino": "output/feature_cache",
          "clip": "output/feature_cache_clip",
          "dinov2": "output/feature_cache_dinov2"}
K_SIGMA = 1.5
KNN = 5
TL = 10


def row_key(r):
    return (r["dataset"], r["detector"], r["method"],
            str(r["contamination_rate"]), str(r["seed"]), str(r["train_limit"]))


def get_completed(p: Path):
    done = set()
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                done.add(row_key(r))
    return done


def save_row(row, p: Path):
    exists = p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--sleep", type=int, default=5)
    ap.add_argument("--max-datasets", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
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
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest

    out_dir = Path("output/exp_p5_assurance")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    status_path = Path("exp_p5_status.txt")

    def status(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        try:
            with open(status_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    datasets = ["mvtec_AD/bottle"] if args.smoke else list(E.ALL_DATASETS)
    seeds = [0] if args.smoke else E.SEEDS

    completed = get_completed(csv_path)
    status("=" * 60)
    status(f"START p5 assurance | priority={prio} threads={args.threads} "
           f"smoke={args.smoke} | done_rows={len(completed)}")

    # grid of (detector, method, c) rows produced per (dataset, seed)
    def cell_rows():
        rows = []
        for c in [0.0, 0.3]:
            rows += [("dino", "audit_remove", c), ("dino", "audit_recapture", c)]
        rows += [("dino", "oracle_remove", 0.3)]
        for b in ["base_knn", "base_lof", "base_iforest"]:
            rows += [("dino", b, 0.3)]
        for det in ["clip", "dinov2"]:
            rows += [(det, "audit_remove", 0.3), (det, "oracle_remove", 0.3)]
        return rows

    def mad_mask(scores):
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        if mad < 1e-12:
            return np.zeros(len(scores), dtype=bool)
        mask = scores > med + K_SIGMA * 1.4826 * mad
        cap = int(0.4 * len(scores))
        if mask.sum() > cap:
            order = np.argsort(scores)[::-1]
            mask = np.zeros(len(scores), dtype=bool)
            mask[order[:cap]] = True
        return mask

    def load_cache(backbone, ds):
        E.CACHE_DIR = Path(CACHES[backbone])
        return E.load_cached_dataset(ds)

    processed = 0
    for ds_idx, ds in enumerate(datasets):
        todo = any(
            (ds, det, meth, str(c), str(s), str(TL)) not in completed
            for s in seeds for det, meth, c in cell_rows())
        if not todo:
            status(f"[{ds_idx+1}/{len(datasets)}] {ds}: complete, skipping")
            continue
        status(f"[{ds_idx+1}/{len(datasets)}] {ds}: running")
        t_ds = time.time()

        # ---------- index-level plan per (seed, c): who is contaminated ----------
        # PASS 1 (dino): decisions + dino-detector evals
        try:
            cached = load_cache("dino", ds)
        except FileNotFoundError as e:
            status(f"    [ERROR] dino cache: {e}")
            continue

        n_all = cached["all_train_features"].shape[0]
        test_labels = cached["test_labels"]
        anomaly_idx = np.where(test_labels == 1)[0]
        normal_idx = np.where(test_labels == 0)[0]

        plans = {}      # (seed, c) -> dict with indices and masks
        for seed in seeds:
            rng = np.random.RandomState(seed)
            sel = sorted(rng.choice(n_all, TL, replace=False)) if n_all > TL \
                else list(range(n_all))
            arr = anomaly_idx.copy()
            np.random.RandomState(seed + 5000).shuffle(arr)
            pool_a = arr[: len(arr) // 2]
            pool_b = arr[len(arr) // 2:]
            eval_idx = np.sort(np.concatenate([normal_idx, pool_b]))
            unused = [i for i in range(n_all) if i not in set(sel)]

            for c in [0.0, 0.3]:
                K = int(round(TL * c))
                is_cont = np.zeros(TL, dtype=bool)
                slot_src = [("train", i) for i in sel]
                if K > 0 and len(pool_a) > 0:
                    rng2 = np.random.RandomState(seed + 2000)
                    repl = rng2.choice(TL, K, replace=False)
                    pick = rng2.choice(len(pool_a), K, replace=len(pool_a) < K)
                    for r_i, p_i in zip(repl, pick):
                        slot_src[r_i] = ("test", int(pool_a[p_i]))
                        is_cont[r_i] = True
                # recapture replacements: deterministic unused normals
                rng3 = np.random.RandomState(seed + 7000)
                order = rng3.permutation(len(unused))
                plans[(seed, c)] = dict(sel=sel, slot_src=slot_src,
                                        is_cont=is_cont, eval_idx=eval_idx,
                                        repl_pool=[unused[i] for i in order])

        def build_bank(cached_b, plan):
            feats = []
            for kind, i in plan["slot_src"]:
                feats.append(cached_b["all_train_features"][i] if kind == "train"
                             else cached_b["test_features"][i])
            return np.stack(feats)

        def eval_bank(cached_b, bank3d, keep_mask, plan, extra=None):
            kept = bank3d[keep_mask]
            if extra is not None and len(extra):
                kept = np.concatenate([kept, extra], axis=0)
            bank = kept.reshape(-1, kept.shape[-1])
            ef = cached_b["test_features"][plan["eval_idx"]]
            el = cached_b["test_labels"][plan["eval_idx"]]
            scores = E.compute_knn_scores(bank, ef, k=KNN)
            return E.compute_metrics(el, scores)

        def emit(det, meth, c, seed, plan, mask, metrics, t0, topm=None):
            is_cont = plan["is_cont"]
            row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset": ds, "setup": "B", "detector": det,
                "auditor": "dino_imgmax_mad1.5" if meth.startswith("audit") or
                           meth.startswith("base") else "",
                "method": meth, "contamination_rate": c, "seed": seed,
                "train_limit": TL, "n_train": TL,
                "n_contaminated": int(is_cont.sum()),
                "n_removed": int(mask.sum()),
                "n_cont_removed": int((mask & is_cont).sum()),
                "top1_cont": "" if topm is None else topm[0],
                "top2_cont": "" if topm is None else topm[1],
                "top3_cont": "" if topm is None else topm[2],
                "img_auroc": round(metrics["img_auroc"], 6),
                "img_aupr": round(metrics["img_aupr"], 6),
                "tpr_fpr1": round(metrics["tpr_fpr1"], 6),
                "tpr_fpr5": round(metrics["tpr_fpr5"], 6),
                "elapsed_s": round(time.time() - t0, 2),
            }
            save_row(row, csv_path)
            completed.add(row_key(row))
            print(f"    {det}/{meth} c={c} seed={seed}: AUROC={row['img_auroc']:.4f} "
                  f"rm={row['n_removed']} (cont {row['n_cont_removed']}/{row['n_contaminated']})",
                  flush=True)

        audit_masks = {}   # (seed, c) -> auditor mask (dino)
        for seed in seeds:
            for c in [0.0, 0.3]:
                plan = plans[(seed, c)]
                need = [m for m in ["audit_remove", "audit_recapture"]
                        if (ds, "dino", m, str(c), str(seed), str(TL)) not in completed]
                need_oracle = c == 0.3 and \
                    (ds, "dino", "oracle_remove", "0.3", str(seed), str(TL)) not in completed
                need_base = c == 0.3 and any(
                    (ds, "dino", b, "0.3", str(seed), str(TL)) not in completed
                    for b in ["base_knn", "base_lof", "base_iforest"])
                # masks are needed for later passes even if dino rows are done
                bank = build_bank(cached, plan)
                ps, _ = E.compute_loo_consistency(bank, k=KNN)
                S = ps.max(axis=1)
                mask = mad_mask(S)
                audit_masks[(seed, c)] = mask
                rank = np.argsort(S)[::-1]
                topm = [int(plan["is_cont"][rank[:m]].sum()) for m in (1, 2, 3)]

                if need or need_oracle or need_base:
                    if "audit_remove" in need:
                        t0 = time.time()
                        m = eval_bank(cached, bank, ~mask, plan)
                        emit("dino", "audit_remove", c, seed, plan, mask, m, t0, topm)
                    if "audit_recapture" in need:
                        t0 = time.time()
                        nrep = int(mask.sum())
                        repl = plan["repl_pool"][:nrep]
                        extra = cached["all_train_features"][repl] if nrep else None
                        m = eval_bank(cached, bank, ~mask, plan, extra)
                        emit("dino", "audit_recapture", c, seed, plan, mask, m, t0, topm)
                    if need_oracle:
                        t0 = time.time()
                        m = eval_bank(cached, bank, ~plan["is_cont"], plan)
                        emit("dino", "oracle_remove", c, seed, plan,
                             plan["is_cont"], m, t0)
                    if need_base:
                        gemb = bank.mean(axis=1)   # (TL, d) global embeddings
                        d2 = ((gemb[:, None, :] - gemb[None, :, :]) ** 2).sum(-1) ** 0.5
                        np.fill_diagonal(d2, np.nan)
                        s_knn = np.nanmean(np.sort(d2, axis=1)[:, :3], axis=1)
                        lof = LocalOutlierFactor(n_neighbors=3)
                        lof.fit(gemb)
                        s_lof = -lof.negative_outlier_factor_
                        iforest = IsolationForest(random_state=seed, n_estimators=100)
                        s_if = -iforest.fit(gemb).score_samples(gemb)
                        for bname, sc in [("base_knn", s_knn), ("base_lof", s_lof),
                                          ("base_iforest", s_if)]:
                            key = (ds, "dino", bname, "0.3", str(seed), str(TL))
                            if key in completed:
                                continue
                            t0 = time.time()
                            bm = mad_mask(sc)
                            m = eval_bank(cached, bank, ~bm, plan)
                            emit("dino", bname, c, seed, plan, bm, m, t0)
        del cached
        import gc
        gc.collect()

        # PASS 2/3: other detectors with the dino auditor masks
        for det in ["clip", "dinov2"]:
            rows_needed = any(
                (ds, det, m, "0.3", str(s), str(TL)) not in completed
                for s in seeds for m in ["audit_remove", "oracle_remove"])
            if not rows_needed:
                continue
            try:
                cb = load_cache(det, ds)
            except FileNotFoundError as e:
                status(f"    [WARN] {det} cache missing: {e}")
                continue
            for seed in seeds:
                plan = plans[(seed, 0.3)]
                mask = audit_masks[(seed, 0.3)]
                bank = build_bank(cb, plan)
                if (ds, det, "audit_remove", "0.3", str(seed), str(TL)) not in completed:
                    t0 = time.time()
                    m = eval_bank(cb, bank, ~mask, plan)
                    emit(det, "audit_remove", 0.3, seed, plan, mask, m, t0)
                if (ds, det, "oracle_remove", "0.3", str(seed), str(TL)) not in completed:
                    t0 = time.time()
                    m = eval_bank(cb, bank, ~plan["is_cont"], plan)
                    emit(det, "oracle_remove", 0.3, seed, plan, plan["is_cont"], m, t0)
            del cb
            gc.collect()

        status(f"[{ds_idx+1}/{len(datasets)}] {ds}: done in {(time.time()-t_ds)/60:.1f} min")
        processed += 1
        if args.max_datasets and processed >= args.max_datasets:
            status("max-datasets reached, stopping (resumable)")
            break
        if not args.smoke:
            time.sleep(args.sleep)

    status(f"END p5 assurance | datasets processed: {processed}")


if __name__ == "__main__":
    main()
