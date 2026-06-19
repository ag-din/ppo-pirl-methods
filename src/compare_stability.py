# =============================================================================
# compare_stability.py
# =============================================================================
#
# Outcome stability is assessed from final evaluation results:
#      - SeedStd : std of per-seed mean returns  → cross-seed reproducibility
#      - SeedIQR : IQR of per-seed mean returns  → robustness to outlier seeds
#      - ΔSeedStd = (SeedStd_PPO - SeedStd_method) / |SeedStd_PPO|
#      - ΔSeedIQR = (SeedIQR_PPO - SeedIQR_method) / |SeedIQR_PPO|
#
#   Positive Δ = more stable than PPO baseline.
#
# Classification (applied independently to each metric):
#   More stable : Δ > 0 AND IQM_method >= IQM_PPO
#   Trade-off   : Δ > 0 BUT IQM_method <  IQM_PPO or vice versa
#   Less stable : Δ < 0
#
# Outputs:
#   Printed table with SeedStd, SeedIQR, their deltas vs PPO, and classifications.
#
# Usage:
#   python compare_stability.py --env CartPole-v1
#   python compare_stability.py --env Reacher-v5 --experiments-dir experiments
#
# =============================================================================

# ── Libraries ──────────────────────────────────────────────────────────-----
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

METHODS = ["RL_PURE", "PIRL_REWARD", "PIRL_STATE", "PIRL_POLICY"]
SEEDS   = list(range(11, 19))   

METHOD_COLORS = {
    "RL_PURE":     "#2C3E50",
    "PIRL_REWARD": "#2980B9",
    "PIRL_STATE":  "#12A650",
    "PIRL_POLICY": "#8E44AD",
}

METHOD_LABELS = {
    "RL_PURE":     "PPO (baseline)",
    "PIRL_REWARD": "PIRL — Reward",
    "PIRL_STATE":  "PIRL — State",
    "PIRL_POLICY": "PIRL — Policy",
}

CLF_COLORS = {
    "more stable": "#27AE60",
    "trade-off":   "#E67E22",
    "less stable": "#E74C3C",
    "baseline":    "#2C3E50",
}


# =============================================================================
# ARG PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3 outcome stability analysis.")
    parser.add_argument("--env",             type=str, required=True)
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rliable_summary(env_dir: Path, method: str) -> dict | None:
    path = env_dir / method / "evaluation" / "data" / "rliable_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def classify(delta: float, iqm_method: float, iqm_ppo: float) -> str:
    if delta > 0 and iqm_method >= iqm_ppo:
        return "more stable"
    elif delta > 0 and iqm_method < iqm_ppo:
        return "trade-off"
    else:
        return "less stable"


def safe_delta(ref: float, method: float) -> float | None:
    if abs(ref) > 1e-9:
        return (ref - method) / abs(ref)
    return None


def fmt_pct(val: float | None) -> str:
    return f"{val*100:+.2f}%" if val is not None else "N/A"


# =============================================================================
# CROSS-SEED STABILITY COMPUTATION
# =============================================================================

def compute_rq3(
    available:         list[str],
    rliable_summaries: dict[str, dict],
) -> list[dict]:

    # ── PPO reference values ──────────────────────────────────────────────────
    ppo_s       = rliable_summaries.get("RL_PURE", {})
    iqm_ppo     = float(ppo_s.get("iqm_raw",           0.0))
    seedstd_ppo = float(ppo_s.get("std_of_seed_means",  0.0))
    seediqr_ppo = float(ppo_s.get("iqr_of_seed_means",  0.0))

    results = []
    for method in available:
        s         = rliable_summaries.get(method, {})
        iqm_m     = float(s.get("iqm_raw",           0.0))
        seedstd_m = float(s.get("std_of_seed_means",  0.0))
        seediqr_m = float(s.get("iqr_of_seed_means",  0.0))

        if method == "RL_PURE":
            results.append({
                "method":            method,
                "iqm_raw":           round(iqm_m,     2),
                "seedstd":           round(seedstd_m, 4),
                "seediqr":           round(seediqr_m, 4),
                "delta_seedstd":     None,
                "delta_seedstd_pct": "—",
                "clf_seedstd":       "baseline",
                "delta_seediqr":     None,
                "delta_seediqr_pct": "—",
                "clf_seediqr":       "baseline",
                "rank_seedstd":      "—",
                "rank_seediqr":      "—",
            })
        else:
            d_seedstd = safe_delta(seedstd_ppo, seedstd_m)
            d_seediqr = safe_delta(seediqr_ppo, seediqr_m)

            results.append({
                "method":            method,
                "iqm_raw":           round(iqm_m,     2),
                "seedstd":           round(seedstd_m, 4),
                "seediqr":           round(seediqr_m, 4),
                "delta_seedstd":     round(d_seedstd, 4) if d_seedstd is not None else None,
                "delta_seedstd_pct": fmt_pct(d_seedstd),
                "clf_seedstd":       classify(d_seedstd or 0, iqm_m, iqm_ppo),
                "delta_seediqr":     round(d_seediqr, 4) if d_seediqr is not None else None,
                "delta_seediqr_pct": fmt_pct(d_seediqr),
                "clf_seediqr":       classify(d_seediqr or 0, iqm_m, iqm_ppo),
                "rank_seedstd":      None,   # filled below
                "rank_seediqr":      None,
            })

    # ── Rankings (PIRL only) ──────────────────────────────────────────────────
    pirl = [r for r in results if r["method"] != "RL_PURE"]
    for i, r in enumerate(sorted(pirl, key=lambda x: x["seedstd"]), 1):
        r["rank_seedstd"] = i
    for i, r in enumerate(sorted(pirl, key=lambda x: x["seediqr"]), 1):
        r["rank_seediqr"] = i

    ppo_r = next(r for r in results if r["method"] == "RL_PURE")
    return [ppo_r] + sorted(
        pirl,
        key=lambda r: r["delta_seediqr"] if r["delta_seediqr"] is not None else float("-inf"),
        reverse=True,
    )


# =============================================================================
# OUTPUT
# =============================================================================

def print_rq3_table(env_name: str, results: list[dict]) -> None:
    w = 100
    print(f"\n{'─'*w}")
    print(f"  Cross-Seed Stability: {env_name}")
    print(f"{'─'*w}")
    print(f"  {'Method':<22} {'SeedStd':>8} {'ΔSeedStd':>10}"
          f"{'SeedIQR':>8} {'ΔSeedIQR':>10}")
    print(f"  {'─'*(w-2)}")
    for r in results:
        label = METHOD_LABELS.get(r["method"], r["method"])
        print(
            f"  {label:<22} "
            f"{r['seedstd']:>8.4f} {r['delta_seedstd_pct']:>10}"
            f"{r['seediqr']:>8.4f} {r['delta_seediqr_pct']:>10}"
        )
    print(f"{'─'*w}\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args     = parse_args()
    env_name = args.env
    env_dir  = Path(args.experiments_dir) / env_name

    if not env_dir.exists():
        raise FileNotFoundError(f"Not found: {env_dir}")

    analysis_dir = env_dir / "analysis"
    plots_dir    = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Cross-seed Stability Analysis: {env_name}")
    print(f"  Output: {analysis_dir}")
    print(f"{'='*55}\n")

    # ── Load evaluation summaries ─────────────────────────────────────────────
    rliable_summaries = {}
    available         = []
    for method in METHODS:
        s = load_rliable_summary(env_dir, method)
        if s is None:
            print(f"  [SKIP] {method} — rliable_summary.json not found")
            continue
        rliable_summaries[method] = s
        available.append(method)
        print(f"  → {method} ✓")

    if "RL_PURE" not in available:
        raise ValueError("PPO baseline not found.")

    # ── Compute ───────────────────────────────────────────────────────────────
    results = compute_rq3(available, rliable_summaries)

    # ── Output ────────────────────────────────────────────────────────────────
    print_rq3_table(env_name, results)

    print(f"{'='*55}")
    print(f"  Cross-seed stability complete: {env_name}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()