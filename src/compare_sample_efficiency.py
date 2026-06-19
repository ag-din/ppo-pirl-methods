# =============================================================================
# compare_sample_efficiency.py
# =============================================================================
#
# For a given environment, loads training summaries and aggregated learning
# curves per method and produces:
#
#   1. Sample Efficiency Table (stdout) — IQM AUC, ΔAUC, rank per method
#   2. learning_curves.png — single-environment Median ± IQR learning
#      curves (combined-figure style), saved to the environment's plots/ dir.
#
# Point estimate: IQM (Interquartile Mean) of per-seed AUC values.
# IQM = mean of middle 50% of seeds — robust to failed seeds, uses more
# data than median. Recommended by Agarwal et al. (2021) — Rliable.
# ΔAUC = (IQM_method - IQM_PPO) / |IQM_PPO|
#
# Usage:
#   python compare_sample_efficiency.py --env Reacher-v5
#   python compare_sample_efficiency.py --env Reacher-v5 --experiments-dir experiments
#
# =============================================================================

# ── Libraries ──────────────────────────────────────────────────────────
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

# Score thresholds per environment (IQR bands and medians clipped to these)
ENV_SCORE_BOUNDS = {
    "Reacher-v5":               (-14.0,    0.0),
    "Acrobot-v1":               (-500.0,   0.0),
    "CartPole-v1":              (   0.0, 500.0),
    "MountainCarContinuous-v0": (  -1.0, 100.0),
}

METHOD_COLORS = {
    "RL_PURE":     "#2C3E50",
    "PIRL_REWARD": "#2980B9",
    "PIRL_STATE":  "#12A650",
    "PIRL_POLICY": "#8E44AD",
}

METHOD_LABELS = {
    "RL_PURE":     "RL_PURE (PPO baseline)",
    "PIRL_REWARD": "PIRL_REWARD",
    "PIRL_STATE":  "PIRL_STATE",
    "PIRL_POLICY": "PIRL_POLICY",
}

METHOD_LINESTYLES = {
    "RL_PURE":     "-",
    "PIRL_REWARD": "-",
    "PIRL_STATE":  "-",
    "PIRL_POLICY": "-",
}

METHOD_ZORDER = {
    "RL_PURE":     5,
    "PIRL_REWARD": 4,
    "PIRL_STATE":  3,
    "PIRL_POLICY": 1,
}


# =============================================================================
# ARG PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Efficiency Analysis for one environment.")
    parser.add_argument("--env",             type=str, required=True,
                        help="Environment name (e.g. Reacher-v5)")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Root experiments directory (default: experiments/)")
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_summary(env_dir: Path, method: str) -> dict | None:
    """Load training summary.json — contains pre-computed AUC values."""
    path = env_dir / method / "training" / "training_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_aggregated_curves(env_dir: Path, method: str) -> dict | None:
    """Load aggregated learning curves from training/training_results_aggregated.jsonl."""
    path = env_dir / method / "training" / "training_results_aggregated.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        return json.loads(f.readline())


# =============================================================================
# IQM
# =============================================================================

def iqm(values: list[float]) -> float:
    """
    Interquartile Mean: mean of the middle 50% of values.
    Trims the bottom and top 25% before averaging.
    Recommended by Agarwal et al. (2021) for RL evaluation with few seeds.
    """
    arr    = np.sort(np.array(values, dtype=float))
    n      = len(arr)
    lo_idx = int(np.floor(0.25 * n))
    hi_idx = int(np.ceil(0.75 * n))
    return round(float(np.mean(arr[lo_idx:hi_idx])), 4)


# =============================================================================
# BOOTSTRAP IQM CI
# =============================================================================

def bootstrap_iqm_ci(
    seed_aucs:   list[float],
    n_bootstrap: int   = 10_000,
    ci:          float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for IQM AUC over seeds.
    Resamples seeds with replacement and computes IQM each time.
    """
    rng   = np.random.default_rng(42)
    boots = [
        iqm(rng.choice(seed_aucs, size=len(seed_aucs), replace=True).tolist())
        for _ in range(n_bootstrap)
    ]
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return round(lo, 4), round(hi, 4)


def bootstrap_delta_iqm_ci(
    seed_aucs_method:   list[float],
    seed_aucs_baseline: list[float],
    n_bootstrap:        int   = 10_000,
    ci:                 float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for ΔAUC = (IQM_method - IQM_baseline) / |IQM_baseline|.
    Resamples both method and baseline seeds independently.
    """
    rng   = np.random.default_rng(42)
    boots = []
    for _ in range(n_bootstrap):
        iqm_m = iqm(rng.choice(seed_aucs_method,   size=len(seed_aucs_method),   replace=True).tolist())
        iqm_b = iqm(rng.choice(seed_aucs_baseline, size=len(seed_aucs_baseline), replace=True).tolist())
        denom = abs(iqm_b)
        boots.append((iqm_m - iqm_b) / denom if denom > 1e-9 else 0.0)

    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return round(lo, 4), round(hi, 4)


def classify_dauc(lo: float, hi: float) -> str:
    """
    Classify ΔAUC based on CI overlap with zero.
    [+] CI entirely above 0 → more efficient
    [~] CI overlaps 0       → no reliable difference
    [−] CI entirely below 0 → less efficient
    """
    if lo > 0:
        return "[+]"
    elif hi < 0:
        return "[−]"
    else:
        return "[~]"


# =============================================================================
# RQ2 COMPUTATION
# =============================================================================

def compute_rq2(
    methods:         list[str],
    train_summaries: dict[str, dict],
    n_bootstrap:     int = 10_000,
) -> list[dict]:

    ppo_s         = train_summaries.get("RL_PURE", {})
    seed_aucs_ppo = ppo_s.get("auc_median_per_seed", [])

    # Point estimate = IQM of per-seed AUCs
    iqm_ppo = iqm(seed_aucs_ppo)

    # Bootstrap CI for PPO IQM
    ppo_ci_lo, ppo_ci_hi = bootstrap_iqm_ci(seed_aucs_ppo, n_bootstrap)

    results = []
    for method in methods:
        s         = train_summaries.get(method, {})
        seed_aucs = s.get("auc_median_per_seed", [])

        # Point estimate = IQM of per-seed AUCs
        auc_iqm = iqm(seed_aucs)

        # IQM CI via bootstrap
        ci_lo, ci_hi = bootstrap_iqm_ci(seed_aucs, n_bootstrap)

        if method == "RL_PURE":
            delta_auc     = None
            delta_auc_pct = "—"
            d_ci_lo       = None
            d_ci_hi       = None
            clf           = "—"

        elif abs(iqm_ppo) > 1e-9 and seed_aucs_ppo:
            delta_auc     = (auc_iqm - iqm_ppo) / abs(iqm_ppo)
            delta_auc_pct = f"{delta_auc*100:+.2f}%"

            # ΔAUC CI via bootstrap
            d_ci_lo, d_ci_hi = bootstrap_delta_iqm_ci(
                seed_aucs, seed_aucs_ppo, n_bootstrap)
            clf = classify_dauc(d_ci_lo, d_ci_hi)

        else:
            delta_auc     = None
            delta_auc_pct = "N/A"
            d_ci_lo       = None
            d_ci_hi       = None
            clf           = "—"

        results.append({
            "method":            method,
            "auc_iqm":           auc_iqm,
            "auc_ci_low":        ci_lo,
            "auc_ci_high":       ci_hi,
            "delta_auc":         round(delta_auc, 4) if delta_auc is not None else None,
            "delta_auc_pct":     delta_auc_pct,
            "delta_auc_ci_low":  round(d_ci_lo * 100, 2) if d_ci_lo is not None else None,
            "delta_auc_ci_high": round(d_ci_hi * 100, 2) if d_ci_hi is not None else None,
            "classification":    clf,
            "n_seeds":           len(s.get("seeds_training", [])),
            "timesteps":         s.get("timesteps_training"),
        })

    # Rank PIRL by ΔAUC descending
    pirl = [r for r in results if r["method"] != "RL_PURE"]
    for rank, r in enumerate(
            sorted(pirl, key=lambda r: r["delta_auc"] or float("-inf"), reverse=True),
            start=1):
        r["rank"] = rank

    ppo_result = next(r for r in results if r["method"] == "RL_PURE")
    ppo_result["rank"] = None

    return [ppo_result] + sorted(
        pirl, key=lambda r: r["delta_auc"] or float("-inf"), reverse=True)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_rq2_learning_curves(
    env_name:    str,
    methods:     list[str],
    curve_data:  dict[str, dict],
    rq2_results: list[dict],
    plots_dir:   Path,
) -> None:
    """
    Single-environment Median ± IQR learning curves, in the style of the
    combined figure: thick lines, IQR bands clipped to environment-specific
    score bounds, and a shared legend below the axes.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    score_min, score_max = ENV_SCORE_BOUNDS.get(env_name, (None, None))

    dauc_lookup    = {r["method"]: r["delta_auc_pct"] for r in rq2_results}
    legend_handles = []
    legend_labels  = []

    for method in methods:
        curves = curve_data.get(method)
        if not curves:
            continue

        median_curve = curves.get("aggregated_median_curve", [])
        if not median_curve:
            continue

        ts      = np.array([p["timestep"] for p in median_curve])
        medians = np.array([p["median"]   for p in median_curve])
        iqrs    = np.array([p["iqr"]      for p in median_curve])

        # Raw IQR bands
        band_lo = medians - iqrs / 2
        band_hi = medians + iqrs / 2

        # Clip bands and medians to score thresholds
        if score_min is not None:
            band_lo = np.clip(band_lo, score_min, None)
            medians = np.clip(medians, score_min, None)
        if score_max is not None:
            band_hi = np.clip(band_hi, None, score_max)
            medians = np.clip(medians, None, score_max)

        color = METHOD_COLORS[method]
        ls    = METHOD_LINESTYLES[method]
        zo    = METHOD_ZORDER[method]
        lw    = 3 if method == "RL_PURE" else 5

        dauc  = dauc_lookup.get(method, "")
        label = (
            f"{METHOD_LABELS[method]}"
            if method != "RL_PURE"
            else METHOD_LABELS[method]
        )

        line, = ax.plot(
            ts, medians,
            color=color, linewidth=lw,
            linestyle=ls, zorder=zo,
            label=label,
        )
        ax.fill_between(ts, band_lo, band_hi, color=color, alpha=0.18)

        legend_handles.append(line)
        legend_labels.append(label)

    # Y-axis limits to score bounds (with a small margin)
    if score_min is not None and score_max is not None:
        margin = (score_max - score_min) * 0.03
        ax.set_ylim(score_min - margin, score_max + margin)

    ax.set_title(env_name, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Timesteps", fontsize=18)
    ax.set_ylabel("Episode Return", fontsize=18)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{float(x/1000):.0f}k")
    )
    ax.tick_params(labelsize=16)

    # ── Shared legend below the axes ─────────────────────────────────────────
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=len(legend_labels) if legend_labels else 1,
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = (plots_dir / "learning_curves.png").resolve()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Learning curves →  {out}")


# =============================================================================
# OUTPUT
# =============================================================================

def print_rq2_table(env_name: str, results: list[dict]) -> None:
    w = 90
    print(f"\n{'─'*w}")
    print(f"  Learning Efficiency (IQM): {env_name}")
    print(f"{'─'*w}")
    print(f"  {'Method':<22} {'IQM AUC':>10} {'95% CI':>20} "
          f"{'ΔAUC':>10} {'ΔAUC 95% CI':>20} {'Clf.':>5} {'Rank':>5}")
    print(f"  {'─'*(w-2)}")

    for r in results:
        rank  = str(r["rank"]) if r["rank"] is not None else "—"
        label = METHOD_LABELS.get(r["method"], r["method"])
        ci    = f"[{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]"

        if r["delta_auc_ci_low"] is not None:
            dci = f"[{r['delta_auc_ci_low']:+.2f}%, {r['delta_auc_ci_high']:+.2f}%]"
        else:
            dci = "—"

        print(
            f"  {label:<22} {r['auc_iqm']:>10.4f} {ci:>20} "
            f"{r['delta_auc_pct']:>10} {dci:>20} "
            f"{r['classification']:>5} {rank:>5}"
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
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    analysis_dir = env_dir / "analysis"
    plots_dir    = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  RQ2 Analysis: {env_name}")
    print(f"  Output: {analysis_dir}")
    print(f"{'='*55}\n")

    # ── Load training summaries and curves ────────────────────────────────────
    train_summaries = {}
    curve_data      = {}
    available       = []

    for method in METHODS:
        train_s = load_training_summary(env_dir, method)
        curves  = load_aggregated_curves(env_dir, method)

        if train_s is None:
            print(f"  [SKIP] {method} — training summary not found")
            continue
        if curves is None:
            print(f"  [SKIP] {method} — aggregated curves not found")
            continue

        train_summaries[method] = train_s
        curve_data[method]      = curves
        available.append(method)
        print(f"  → {method} ✓")

    if "RL_PURE" not in available:
        raise ValueError("PPO baseline (RL_PURE) results not found. Cannot compute ΔAUC.")

    # ── Compute ΔAUC ───────────────────────────────────────────────────────────
    results = compute_rq2(available, train_summaries)

    # ── Output ────────────────────────────────────────────────────────────────
    print_rq2_table(env_name, results)
    plot_rq2_learning_curves(env_name, available, curve_data, results, plots_dir)

    print(f"{'='*55}")
    print(f"  Sample Efficiency Analysis complete: {env_name}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()