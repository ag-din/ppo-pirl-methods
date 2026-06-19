# =============================================================================
# compare_final_performance.py
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

METHODS = ["RL_PURE", "PIRL_REWARD", "PIRL_STATE", "PIRL_POLICY"]

METHOD_COLORS = {
    "RL_PURE":     "#4F5F6C",
    "PIRL_REWARD": "#3894D1",
    "PIRL_STATE":  "#28BD66",
    "PIRL_POLICY": "#B073CB",
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

CLF_COLORS = {
    "+":        "#27AE60",
    "-":        "#E74C3C",
    "~":        "#7F8C8D",
    "baseline": "#2C3E50",
}


# =============================================================================
# ARG PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Performance Comparison")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Environment name (default: CartPole-v1)")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Root experiments directory (default: experiments/)")
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rliable_summary(env_dir: Path, method: str) -> dict | None:
    """Load pre-computed rliable_summary.json for a method."""
    path = env_dir / method / "evaluation" / "data" / "rliable_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_performance_profile(env_dir: Path, method: str) -> dict | None:
    """Load pre-computed performance_profile.json for a method."""
    path = env_dir / method / "evaluation" / "data" / "performance_profile.json"
    if not path.exists():
        print(f"  [WARN] performance_profile.json not found for {method} - skipping profile plot")
        return None
    return json.loads(path.read_text())


# =============================================================================
# FINAL PERFORMANCE COMPUTATION
# =============================================================================

def classify_ci(
    method_lo: float, method_hi: float,
    ppo_lo:    float, ppo_hi:    float,
    iqm_method: float, iqm_ppo:  float,
) -> str:
    if method_lo > ppo_hi and iqm_method > iqm_ppo:
        return "+"
    elif method_hi < ppo_lo and iqm_method < iqm_ppo:
        return "-"
    else:
        return "~"


def compute_rq1(
    methods:           list,
    rliable_summaries: dict,
) -> list:
    """
    Compute RQ1 metrics for all methods relative to PPO baseline.
    Includes IQM and Mean with 95% CIs.
    DeltaR = (IQM_method - IQM_PPO) / |IQM_PPO|.
    """
    ppo     = rliable_summaries.get("RL_PURE", {})
    iqm_ppo = float(ppo.get("iqm_raw",         0.0))
    ppo_lo  = float(ppo.get("iqm_raw_ci_low",  0.0))
    ppo_hi  = float(ppo.get("iqm_raw_ci_high", 0.0))

    results = []
    for method in methods:
        s      = rliable_summaries.get(method, {})
        iqm    = float(s.get("iqm_raw",         0.0))
        ci_lo  = float(s.get("iqm_raw_ci_low",  0.0))
        ci_hi  = float(s.get("iqm_raw_ci_high", 0.0))

        mean_raw       = float(s.get("mean_raw",          0.0))
        mean_raw_ci_lo = float(s.get("mean_raw_ci_low",   0.0))
        mean_raw_ci_hi = float(s.get("mean_raw_ci_high",  0.0))

        if abs(iqm_ppo) > 1e-9:
            delta_r     = (iqm - iqm_ppo) / abs(iqm_ppo)
            delta_r_pct = f"{delta_r*100:+.2f}%"
        else:
            delta_r     = None
            delta_r_pct = "N/A (PPO ~ 0)"

        if method == "RL_PURE":
            classification = "baseline"
        else:
            classification = classify_ci(ci_lo, ci_hi, ppo_lo, ppo_hi, iqm, iqm_ppo)

        results.append({
            "method":              method,
            "iqm_raw":             round(iqm,   8),
            "iqm_ci_low":          round(ci_lo, 8),
            "iqm_ci_high":         round(ci_hi, 8),
            "mean_raw":            round(mean_raw,       8),
            "mean_raw_ci_low":     round(mean_raw_ci_lo, 8),
            "mean_raw_ci_high":    round(mean_raw_ci_hi, 8),
            "delta_r":             round(delta_r, 8) if delta_r is not None else None,
            "delta_r_pct":         delta_r_pct,
            "classification":      classification,
            "median_raw":          round(float(s.get("median_raw",        0.0)), 8),
            "median_raw_ci_low":   round(float(s.get("median_raw_ci_low", 0.0)), 8),
            "median_raw_ci_high":  round(float(s.get("median_raw_ci_high",0.0)), 8),
            "std_of_seed_means":   round(float(s.get("std_of_seed_means", 0.0)), 8),
            "n_seeds":             s.get("n_seeds"),
            "n_eval_episodes":     s.get("n_eval_episodes"),
        })

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_performance_profiles(
    env_name:  str,
    env_dir:   Path,
    available: list,
    plots_dir: Path,
) -> None:
    """
    Single-environment performance profiles: load the pre-saved
    performance_profile.json for each method and plot all curves on one
    axes, with shaded 95% CI bands and a shared legend below.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    legend_handles = []
    legend_labels  = []
    any_plotted    = False

    for method in METHODS:
        if method not in available:
            continue

        profile = load_performance_profile(env_dir, method)
        if profile is None:
            continue

        thresholds = np.array(profile["thresholds"])
        scores     = np.array(profile["score_distribution"])
        ci_low     = np.array(profile["score_distribution_ci_low"])
        ci_high    = np.array(profile["score_distribution_ci_high"])

        color = METHOD_COLORS[method]
        ls    = METHOD_LINESTYLES[method]
        zo    = METHOD_ZORDER[method]
        lw    = 3 if method == "RL_PURE" else 5

        line, = ax.plot(
            thresholds, scores,
            color=color, linewidth=lw,
            linestyle=ls, zorder=zo,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(thresholds, ci_low, ci_high, color=color, alpha=0.6)
        any_plotted = True

        legend_handles.append(line)
        legend_labels.append(METHOD_LABELS[method])

    if not any_plotted:
        print(f"  [SKIP] {env_name} - no performance profile data found")
        plt.close(fig)
        return

    ax.set_title(env_name, fontsize=20, fontweight="bold", pad=10)
    ax.set_xlabel("Normalized Score tau", fontsize=19)
    ax.set_ylabel("Fraction of runs with score > tau", fontsize=19)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=18)

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = (plots_dir / "performance_profiles.png").resolve()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Performance profiles ->  {out}")


# =============================================================================
# OUTPUT
# =============================================================================

def print_rq1_table(env_name: str, results: list) -> None:
    """Print results as a formatted table to stdout."""
    w = 100
    print(f"\n{'-'*w}")
    print(f"  Final Performance: {env_name}")
    print(f"{'-'*w}")
    print(f"  {'Method':<22} {'IQM':>8} {'IQM 95% CI':>22} {'Mean':>8} {'Mean 95% CI':>22} {'DeltaR':>9} {'':>4}")
    print(f"  {'-'*(w-2)}")
    for r in results:
        iqm_ci   = f"[{r['iqm_ci_low']:.1f}, {r['iqm_ci_high']:.1f}]"
        mean_ci  = f"[{r['mean_raw_ci_low']:.1f}, {r['mean_raw_ci_high']:.1f}]"
        dr_str   = r["delta_r_pct"] if r["method"] != "RL_PURE" else "-"
        clf      = r["classification"]
        label    = METHOD_LABELS.get(r["method"], r["method"])
        clf_tag  = f"[{clf}]" if r["method"] != "RL_PURE" else ""
        print(
            f"  {label:<22} {r['iqm_raw']:>8.1f} {iqm_ci:>22} "
            f"{r['mean_raw']:>8.1f} {mean_ci:>22} {dr_str:>9}  {clf_tag}"
        )
    print(f"{'-'*w}\n")


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
    print(f"  Final Performance Analysis: {env_name}")
    print(f"  Output: {analysis_dir}")
    print(f"{'='*55}\n")

    rliable_summaries = {}
    available = []

    for method in METHODS:
        s = load_rliable_summary(env_dir, method)
        if s is None:
            print(f"  [SKIP] {method} - rliable_summary.json not found")
            continue
        rliable_summaries[method] = s
        available.append(method)
        print(f"  -> {method} OK")

    if "RL_PURE" not in available:
        raise ValueError("PPO baseline (RL_PURE) results not found. Cannot compute Final Performance.")

    results = compute_rq1(available, rliable_summaries)

    print_rq1_table(env_name, results)
    plot_performance_profiles(env_name, env_dir, available, plots_dir)

    print(f"{'='*55}")
    print(f"  Final performance analysis complete: {env_name}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()