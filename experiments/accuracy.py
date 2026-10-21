"""Accuracy experiment (paper Table 3) on Abalone: DisTFD against TANE.

For each seed, compare DisTFD's minimal FDs (|LHS| <= max_lhs) with Desbordante
TANE and write precision / recall / F1 to accuracy.csv in the output directory
(default results/abalone/). A sampling-ratio sweep goes to sampling_sweep.csv.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distfd.distfd import Config, discover  # noqa: E402
from distfd.encode import load  # noqa: E402
from distfd.metrics import prf  # noqa: E402
from distfd.spark import get_spark  # noqa: E402
from distfd.truth import tane  # noqa: E402

TABLE = "abalone"
DATA = ROOT / "data" / "abalone" / "abalone.csv"


def named(fds, cols):
    return sorted([sorted(cols[i] for i in x), cols[a]] for x, a in fds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--max-lhs", type=int, default=0, help="0 = no cap (n_cols - 1)")
    ap.add_argument("--sweep", nargs="*", type=float, default=[0.05, 0.10, 0.25],
                    help="sample ratios for the DisTFD sampling sweep")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "abalone", help="output directory")
    args = ap.parse_args()

    out = args.out
    (out / "fds").mkdir(parents=True, exist_ok=True)
    sc = get_spark(4).sparkContext
    cols, codes, df = load(DATA)
    max_lhs = args.max_lhs or codes.shape[1] - 1
    t0 = time.perf_counter()
    truth = tane(df, max_lhs)
    t_truth = time.perf_counter() - t0
    (out / "fds" / f"{TABLE}__TANE.json").write_text(json.dumps(named(truth, cols)))
    print(f"{TABLE}: {codes.shape[0]} x {codes.shape[1]}, |LHS| <= {max_lhs}, TANE {len(truth)} FDs in {t_truth:.1f}s", flush=True)

    rows, sweep = [], []
    for seed in args.seeds:
        cfg = Config(seed=seed, max_lhs=max_lhs)
        fds, stats = discover(sc, codes, cfg)
        m = prf(fds, truth)
        rows.append({"table": TABLE, "rows": codes.shape[0], "cols": codes.shape[1], "max_lhs": max_lhs, "seed": seed,
                     "method": "DisTFD", "n_fds": len(fds), "n_truth": len(truth), **m, "seconds": stats["seconds"]})
        (out / "fds" / f"{TABLE}__DisTFD__seed{seed}.json").write_text(json.dumps(named(fds, cols)))
        (out / "fds" / f"{TABLE}__DisTFD__seed{seed}__stats.json").write_text(json.dumps(stats, indent=1))
        print(f"  seed {seed} DisTFD P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} ({len(fds)} FDs, {stats['seconds']:.1f}s)", flush=True)
        for ratio in args.sweep:
            fds, stats = discover(sc, codes, replace(cfg, sample_ratio=ratio))
            pruned = sum(lv["pruned_by_sample"] for lv in stats["levels"])
            cands = sum(lv["candidates"] for lv in stats["levels"])
            sweep.append({"table": TABLE, "seed": seed, "sample_ratio": ratio,
                          "sample_rows": stats["sample_rows"], "candidates": cands,
                          "pruned_by_sample": pruned, "share_pruned": pruned / cands if cands else 0,
                          **prf(fds, truth), "seconds": stats["seconds"]})
    pd.DataFrame(rows).to_csv(out / "accuracy.csv", index=False)
    pd.DataFrame(sweep).to_csv(out / "sampling_sweep.csv", index=False)


if __name__ == "__main__":
    main()
