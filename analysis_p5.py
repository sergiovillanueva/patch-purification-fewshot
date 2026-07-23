"""
Analysis of the P5 commissioning-time assurance experiments, evaluated against
two acceptance gates that were registered before any result was inspected.

Produces:
  [1] Transfer table per runtime detector (dino / dinov2 / clip):
      contaminated (none) | self-screening | DINOv3 audit | oracle.
      'none' and 'self-screening' come from the P4 CSVs (same banks, verified
      bit-identical construction); 'audit' and 'oracle' from the P5 CSV.
  [2] Recapture vs removal on the DINOv3 detector, plus clean-set cost at c=0.
  [3] Review-budget curve: contaminated images found reviewing top-1/2/3.
  [4] Image-level audit baselines (kNN / LOF / IsolationForest on global
      embeddings) vs the max-LOO auditor.
  [5] GATE verdicts.

Tolerant to partial data (reports n categories used). Run from repo root:
  python analysis_p5.py
"""
import csv
import statistics as st
from collections import defaultdict
from scipy.stats import wilcoxon

P5 = "output/exp_p5_assurance/results.csv"
P4 = {"dino": "output/exp_p4_image_level/results.csv",
      "clip": "output/exp_p4_image_level/results_clip.csv",
      "dinov2": "output/exp_p4_image_level/results_dinov2.csv"}


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def dedup(rows, keyf):
    seen, out = set(), []
    for r in rows:
        k = keyf(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


p5 = dedup(load(P5), lambda r: (r["dataset"], r["detector"], r["method"],
                                r["contamination_rate"], r["seed"], r["train_limit"]))
p4 = {b: dedup(load(p), lambda r: (r["dataset"], r["setup"], r["method"],
                                   str(r["param_k"]), r["contamination_rate"],
                                   r["seed"], r["train_limit"]))
      for b, p in P4.items()}


def perds_p5(det, method, c):
    by = defaultdict(list)
    for r in p5:
        if (r["detector"] == det and r["method"] == method
                and r["contamination_rate"] == str(c) and r["train_limit"] == "10"):
            by[r["dataset"]].append(float(r["img_auroc"]))
    return {d: st.mean(v) for d, v in by.items() if len(v) >= 5}


def perds_p4(det, method, par, c):
    by = defaultdict(list)
    for r in p4[det]:
        if (r["setup"] == "B" and r["method"] == method and str(r["param_k"]) == str(par)
                and r["contamination_rate"] == str(c) and r["train_limit"] == "10"):
            by[r["dataset"]].append(float(r["img_auroc"]))
    return {d: st.mean(v) for d, v in by.items() if len(v) >= 5}


def paired(base, other):
    com = sorted(set(base) & set(other))
    diffs = [other[d] - base[d] for d in com]
    if len(diffs) < 4:
        return com, diffs, float("nan")
    return com, diffs, wilcoxon(diffs).pvalue


print("=" * 72)
print("[1] TRANSFER TABLE (Setup B, N=10, c=0.3; per-category means, 5 seeds)")
gate_a = {}
for det in ["dino", "dinov2", "clip"]:
    none = perds_p4(det, "none", "", 0.3)
    self_scr = perds_p4(det, "imgmax_mad", "1.5", 0.3)
    audit = perds_p5(det, "audit_remove", 0.3)
    oracle = perds_p5(det, "oracle_remove", 0.3)
    com = sorted(set(none) & set(audit) & set(oracle))
    if not com:
        print(f"  {det:8s}: no data yet")
        continue
    dmg_ref = perds_p4(det, "none", "", 0.0)
    comd = sorted(set(dmg_ref) & set(none))
    dmg = st.mean([dmg_ref[d] - none[d] for d in comd]) if comd else float("nan")
    _, dif_a, p_a = paired({d: none[d] for d in com}, {d: audit[d] for d in com})
    _, dif_o, p_o = paired({d: none[d] for d in com}, {d: oracle[d] for d in com})
    selfv = st.mean([self_scr[d] for d in com if d in self_scr]) if self_scr else float("nan")
    print(f"  {det:8s} n={len(com):2d} damage={dmg:+.4f} | none={st.mean([none[d] for d in com]):.4f} "
          f"self={selfv:.4f} | audit={st.mean([audit[d] for d in com]):.4f} "
          f"gain={st.mean(dif_a):+.4f} rec={st.mean(dif_a)/dmg*100 if dmg==dmg and abs(dmg)>1e-9 else float('nan'):.0f}% "
          f"p={p_a:.1e} wins={sum(1 for x in dif_a if x>0)}/{len(dif_a)} | "
          f"oracle gain={st.mean(dif_o):+.4f} (rec {st.mean(dif_o)/dmg*100 if dmg==dmg and abs(dmg)>1e-9 else float('nan'):.0f}%, p={p_o:.1e})")
    gate_a[det] = (st.mean(dif_a), p_a, len(com))

print()
print("[2] RECAPTURE vs REMOVAL (dino detector)")
none3 = perds_p4("dino", "none", "", 0.3)
rem3 = perds_p5("dino", "audit_remove", 0.3)
rec3 = perds_p5("dino", "audit_recapture", 0.3)
com = sorted(set(none3) & set(rem3) & set(rec3))
if com:
    _, dr, pr = paired({d: none3[d] for d in com}, {d: rem3[d] for d in com})
    _, dc, pc = paired({d: none3[d] for d in com}, {d: rec3[d] for d in com})
    _, drc, prc = paired({d: rem3[d] for d in com}, {d: rec3[d] for d in com})
    print(f"  n={len(com)} c=0.3: remove gain={st.mean(dr):+.4f} (p={pr:.1e}) | "
          f"recapture gain={st.mean(dc):+.4f} (p={pc:.1e}) | "
          f"recapture-vs-remove={st.mean(drc):+.4f} (p={prc:.1e})")
none0 = perds_p4("dino", "none", "", 0.0)
rem0 = perds_p5("dino", "audit_remove", 0.0)
rec0 = perds_p5("dino", "audit_recapture", 0.0)
com0 = sorted(set(none0) & set(rem0) & set(rec0))
if com0:
    print(f"  n={len(com0)} c=0.0 CLEAN COST: remove={st.mean([rem0[d]-none0[d] for d in com0]):+.4f} | "
          f"recapture={st.mean([rec0[d]-none0[d] for d in com0]):+.4f}")

print()
print("[3] REVIEW BUDGET (fraction of contaminated found in auditor top-m, c=0.3)")
tops = defaultdict(list)
for r in p5:
    if (r["detector"] == "dino" and r["method"] == "audit_remove"
            and r["contamination_rate"] == "0.3" and r["top1_cont"] not in ("", None)):
        nc = int(r["n_contaminated"])
        if nc > 0:
            for m in (1, 2, 3):
                tops[m].append(int(r[f"top{m}_cont"]) / nc)
for m in (1, 2, 3):
    if tops[m]:
        print(f"  reviewing top-{m} of 10: finds {st.mean(tops[m])*100:.1f}% of contaminated "
              f"(n={len(tops[m])} cells)")

print()
print("[4] AUDIT BASELINES on global embeddings (dino detector, c=0.3)")
for b in ["base_knn", "base_lof", "base_iforest"]:
    v = perds_p5("dino", b, 0.3)
    com = sorted(set(none3) & set(v))
    if not com:
        print(f"  {b:14s}: no data yet")
        continue
    _, d, p = paired({x: none3[x] for x in com}, {x: v[x] for x in com})
    pr_ = [float(r["n_cont_removed"]) / max(1, float(r["n_removed"]))
           for r in p5 if r["detector"] == "dino" and r["method"] == b
           and r["contamination_rate"] == "0.3" and float(r["n_removed"]) > 0]
    print(f"  {b:14s} n={len(com):2d} gain={st.mean(d):+.4f} p={p:.1e} "
          f"removalP={st.mean(pr_):.2f}")

print()
print("=" * 72)
print("[5] PRE-REGISTERED GATES")
ok = [d for d in ("dinov2", "clip") if d in gate_a and gate_a[d][0] > 0 and gate_a[d][1] < 0.05]
if not any(d in gate_a for d in ("dinov2", "clip")):
    print("  GATE-A: insufficient data")
else:
    print(f"  GATE-A (audit transfer, >=1 non-dino detector with gain>0 and p<0.05): "
          f"{'PASS (' + ', '.join(ok) + ')' if ok else 'FAIL'}")
if com and com0:
    d_ok = st.mean(drc) >= -1e-9 or prc > 0.05
    c_ok = abs(st.mean([rec0[d] - none0[d] for d in com0])) <= 0.001
    print(f"  GATE-D (recapture >= removal and clean-set cost |delta|<=0.001): "
          f"{'PASS' if (d_ok and c_ok) else 'FAIL'} "
          f"(recapture-remove={st.mean(drc):+.4f}, clean={st.mean([rec0[d]-none0[d] for d in com0]):+.4f})")
