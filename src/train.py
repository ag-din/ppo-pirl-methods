# =============================================================================
# train.py — Final model training using best hyperparameters from tuning
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# ── Internal ──────────────────────────────────────────────────────────────────
from src.env_utils import make_env, get_eval_method
from src.callbacks import PhysicsPolicyCallback
from src.physics import get_phi


# =============================================================================
# UTILS
# =============================================================================

def get_results_dir() -> Path:
    """Return and create the results directory for this experiment."""
    results_dir = Path(f"results_{cfg["env_name"]}_{cfg["method"]}_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_seed_dir(results_dir: Path, seed: int) -> Path:
    """Return and create the per-seed logs directory."""
    seed_dir = results_dir / f"logs_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def set_global_seeds(seed: int = 42) -> None:
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def linear_schedule(initial_value: float):
    """Linear learning rate schedule decaying to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# =============================================================================
# LOAD HYPERPARAMETERS
# =============================================================================

def load_best_params(cfg: dict) -> tuple[dict, float]:
    """Load best hyperparameters and alpha from tuning results."""

    tuning_dir  = Path(cfg["tuning"]["tuning_dir"])
    params_path = tuning_dir / "best_params.json"

    if not params_path.exists():
        raise FileNotFoundError(
            f"Best params not found at {params_path}. "
            f"Run tuning phase first."
        )

    raw = json.loads(params_path.read_text())

    alpha = float(raw.pop("alpha", 1.0))
    raw["learning_rate"] = linear_schedule(raw["learning_rate"])

    print(f"  Loaded best params from: {params_path}")
    print(f"  alpha = {alpha}")
    return raw, alpha


# =============================================================================
# METRICS
# =============================================================================

def build_mean_curve(timesteps: list, eval_results: list) -> list[dict]:
    """Build mean ± std learning curve from EvalCallback results.

    Each checkpoint entry contains:
        timestep : int   — environment step at evaluation
        mean     : float — mean episode return across eval episodes
        std      : float — standard deviation across eval episodes
    """
    return [
        {
            "timestep": int(t),
            "mean":     round(float(np.mean(r)), 4),
            "std":      round(float(np.std(r)), 4),
        }
        for t, r in zip(timesteps, eval_results)
    ]


def build_median_curve(timesteps: list, eval_results: list) -> list[dict]:
    """Build median ± IQR learning curve from EvalCallback results.

    Each checkpoint entry contains:
        timestep : int   — environment step at evaluation
        median   : float — median episode return across eval episodes
        iqr      : float — interquartile range (Q75 - Q25) across eval episodes
    """
    return [
        {
            "timestep": int(t),
            "median":   round(float(np.median(r)), 4),
            "iqr":      round(float(np.percentile(r, 75) - np.percentile(r, 25)), 4),
        }
        for t, r in zip(timesteps, eval_results)
    ]


def build_aggregated_curves(seed_results: list[dict]) -> dict:
    """Aggregate learning curves across all seeds per checkpoint.

    For mean curve  : aggregates seed-level means   → mean ± std across seeds
    For median curve: aggregates seed-level medians → median ± IQR across seeds
    """
    timesteps = [p["timestep"] for p in seed_results[0]["mean_learning_curve"]]

    agg_mean_curve   = []
    agg_median_curve = []

    for i, t in enumerate(timesteps):

        means_at_t   = [r["mean_learning_curve"][i]["mean"]     for r in seed_results]
        medians_at_t = [r["median_learning_curve"][i]["median"] for r in seed_results]

        agg_mean_curve.append({
            "timestep": t,
            "mean":     round(float(np.mean(means_at_t)), 4),
            "std":      round(float(np.std(means_at_t)), 4),
        })

        agg_median_curve.append({
            "timestep": t,
            "median":   round(float(np.median(medians_at_t)), 4),
            "iqr":      round(float(
                np.percentile(medians_at_t, 75) - np.percentile(medians_at_t, 25)
            ), 4),
        })

    return {
        "aggregated_mean_curve":   agg_mean_curve,
        "aggregated_median_curve": agg_median_curve,
    }


def compute_auc(learning_curve: list[dict], value_key: str) -> float:
    """AUC of a learning curve via trapezoidal integration, normalized by
    total timesteps. Result is in the same unit as the reward.

    Args:
        learning_curve : list of checkpoint dicts (must contain 'timestep')
        value_key      : key to use as the y-axis value ('mean' or 'median')
    """
    timesteps = [p["timestep"] for p in learning_curve]
    values    = [p[value_key] for p in learning_curve]

    auc            = float(np.trapezoid(values, timesteps))
    auc_normalized = auc / timesteps[-1]

    return round(auc_normalized, 4)


def compute_y_limits(seed_results: list[dict]) -> tuple[float, float]:
    """Compute global y-axis limits across all seeds and both curve types.
    Used to keep the same scale across all individual plots.
    """
    all_values = []

    for r in seed_results:
        for p in r["mean_learning_curve"]:
            all_values.append(p["mean"] + p["std"])
            all_values.append(p["mean"] - p["std"])
        for p in r["median_learning_curve"]:
            all_values.append(p["median"] + p["iqr"] / 2)
            all_values.append(p["median"] - p["iqr"] / 2)

    margin = (max(all_values) - min(all_values)) * 0.05
    return min(all_values) - margin, max(all_values) + margin


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


# =============================================================================
# TRAINING
# =============================================================================

def train_seed(
    seed: int,
    best_params: dict,
    alpha: float,
    cfg: dict,
    training_dir: Path,
) -> dict:
    """Train one PPO model for a given seed. Returns per-seed results."""

    env_name        = cfg["env_name"]
    method          = cfg["method"]
    n_envs          = cfg["n_envs"]
    timesteps       = cfg["training"]["timesteps"]
    eval_freq       = cfg["training"]["eval_freq"] // n_envs
    n_eval_episodes = cfg["training"]["n_eval_episodes"]
    eval_seed       = cfg["training"].get("eval_seed", seed + 100)

    seed_dir = get_seed_dir(training_dir, seed)
    vecnorm_path = seed_dir / "vecnormalize.pkl"

    # ── Training env — with PIRL wrapper if applicable ────────────────────────
    train_env = make_env(
        env_name=env_name,
        method=method,
        seed=seed,
        n_envs=n_envs,
        alpha=alpha,
        training=True,
    )

    # ── Eval env — correct wrapper for this method ────────────────────────────
    eval_env = make_env(
        env_name=env_name,
        method=get_eval_method(method),
        seed=eval_seed,
        n_envs=n_envs,
        alpha=alpha,
        training=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(seed_dir),
        log_path=str(seed_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        **best_params,
        verbose=1,
        seed=seed,
    )

    # ── Callbacks — add PhysicsPolicyCallback for PIRL_POLICY ────────────────
    callbacks = [eval_callback]
    if method == "PIRL_POLICY":
        phi_fn = get_phi(env_name)
        callbacks.append(PhysicsPolicyCallback(phi_fn=phi_fn, alpha=alpha))

    seed_start = time.perf_counter()
    model.learn(timesteps, callback=callbacks)
    seed_elapsed = time.perf_counter() - seed_start

    # ── Save VecNormalize stats for evaluation phase ───────────────────────────
    train_env.save(str(vecnorm_path))

    train_env.close()
    eval_env.close()

    ts           = eval_callback.evaluations_timesteps
    eval_results = eval_callback.evaluations_results

    mean_curve   = build_mean_curve(ts, eval_results)
    median_curve = build_median_curve(ts, eval_results)

    # ── AUC ───────────────────────────────────────────────────────────────────
    auc_mean   = compute_auc(mean_curve,   value_key="mean")
    auc_median = compute_auc(median_curve, value_key="median")

    print(
        f"  Seed {seed}: {seed_elapsed:.2f} sec  |  "
        f"AUC mean: {auc_mean:.2f}  |  "
        f"AUC median: {auc_median:.2f}  |  "
        f"model → {str(seed_dir / 'best_model.zip')}"
    )

    return {
        "seed":                  seed,
        "seed_dir":              str(seed_dir),
        "training_time":         round(seed_elapsed, 2),
        "model_path":            str(seed_dir / "best_model.zip"),
        "vecnorm_path":          str(vecnorm_path),
        "auc_mean":              auc_mean,
        "auc_median":            auc_median,
        "mean_learning_curve":   mean_curve,
        "median_learning_curve": median_curve,
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_training_results(seed_results: list[dict], agg_curves: dict, results_dir: Path) -> Path:
    """Save per-seed results (in logs dirs) and aggregated curves (in results_dir)."""

    for record in seed_results:
        seed     = record["seed"]
        seed_dir = Path(record["seed_dir"])
        path     = seed_dir / f"training_results_seed_{seed}.jsonl"
        with path.open("w") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Seed {seed} results  →  {path}")

    agg_path = results_dir / f"training_results_aggregated.jsonl"
    with agg_path.open("w") as f:
        f.write(json.dumps({"seed": "aggregated", **agg_curves}) + "\n")
    print(f"  Aggregated results  →  {agg_path}")

    return agg_path


def save_training_summary(
    cfg: dict,
    seed_results: list[dict],
    agg_curves: dict,
    results_dir: Path,
    elapsed: float,
) -> Path:
    """Save experiment-level training summary to results_dir."""

    aucs_mean   = [r["auc_mean"]   for r in seed_results]
    aucs_median = [r["auc_median"] for r in seed_results]

    auc_agg_mean   = compute_auc(agg_curves["aggregated_mean_curve"],   value_key="mean")
    auc_agg_median = compute_auc(agg_curves["aggregated_median_curve"], value_key="median")

    summary = {
        # Experiment
        "env_name":                    cfg["env_name"],
        "method":                      cfg["method"],

        # Compute
        "n_envs":                      cfg["n_envs"],
        "training_total_time_sec":     round(elapsed, 2),
        "mean_time_per_seed_sec":      round(elapsed / len(cfg["training"]["seeds"]), 2),
        "seed_times_sec":              [r["training_time"] for r in seed_results],

        # Config
        "timesteps_training":          cfg["training"]["timesteps"],
        "seeds_training":              cfg["training"]["seeds"],
        "n_eval_episodes":             cfg["training"]["n_eval_episodes"],
        "eval_freq":                   cfg["training"]["eval_freq"],

        # AUC per seed — mean curve
        "auc_mean_per_seed":           aucs_mean,
        "auc_mean_mean":               round(float(np.mean(aucs_mean)), 4),
        "auc_mean_std":                round(float(np.std(aucs_mean)), 4),
        "auc_mean_median":             round(float(np.median(aucs_mean)), 4),

        # AUC per seed — median curve
        "auc_median_per_seed":         aucs_median,
        "auc_median_mean":             round(float(np.mean(aucs_median)), 4),
        "auc_median_std":              round(float(np.std(aucs_median)), 4),
        "auc_median_median":           round(float(np.median(aucs_median)), 4),

        # AUC aggregated
        "auc_aggregated_mean":         auc_agg_mean,
        "auc_aggregated_median":       auc_agg_median,

        "date":                        datetime.now().isoformat(),
        "sb3_version":                 stable_baselines3.__version__,
        "python_version":              platform.python_version(),
    }

    output_path = results_dir / f"training_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary             →  {output_path}")
    return output_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure as png"""
    fig.savefig(path.with_suffix(".png"), format="png", bbox_inches="tight")


def _plot_curve_panels(
    ax_mean: plt.Axes,
    ax_median: plt.Axes,
    mean_curve: list[dict],
    median_curve: list[dict],
    auc_mean: float,
    auc_median: float,
    y_lim: tuple[float, float],
    title_suffix: str,
    score_min: float = -float("inf"),
    score_max: float =  float("inf"),
) -> None:
    """Shared plotting logic for mean and median panels."""

    ts = [p["timestep"] for p in mean_curve]

    # ── Mean ± std ────────────────────────────────────────────────────────────
    means = [p["mean"] for p in mean_curve]
    stds  = [p["std"]  for p in mean_curve]
    upper = [min(m + i/2, score_max) for m, i in zip(means, stds)]
    lower = [max(m - i/2, score_min) for m, i in zip(means, stds)]
    ax_mean.plot(ts, means, linewidth=2, label="Mean", color="steelblue")
    ax_mean.fill_between(ts,
        lower,
        upper,
        alpha=0.25, label="± Std"
    )
    ax_mean.set_title(f"Mean ± Std", fontsize=12)
    ax_mean.set_xlabel("Timesteps (x 1000)")
    ax_mean.set_ylabel("Episode Return")
    ax_mean.set_ylim(*y_lim)  
    ax_mean.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}"))
    ax_mean.xaxis.set_major_locator(plt.MultipleLocator(5_000))
    ax_mean.tick_params(which='both', width=0.8, direction='in', bottom=True, top=True, left=True, right=True)
    ax_mean.tick_params(which='major', length=3.5)
    ax_mean.tick_params(which='minor', length=3.5)
    ax_mean.grid(False)

    # ── Median ± IQR ──────────────────────────────────────────────────────────
    medians = [p["median"] for p in median_curve]
    iqrs    = [p["iqr"]    for p in median_curve]
    upper = [min(m + i/2, score_max) for m, i in zip(medians, iqrs)]
    lower = [max(m - i/2, score_min) for m, i in zip(medians, iqrs)]
    ax_median.plot(ts, medians, color="steelblue", linewidth=2, label="Median")
    ax_median.fill_between(ts,
        lower,
        upper,
        alpha=0.25, color="steelblue", label="± IQR/2"
    )
    ax_median.set_title(f"Median ± IQR", fontsize=12)
    ax_median.set_xlabel("Timesteps (x 1000)")
    ax_median.set_ylabel("Episode Return")
    ax_median.set_ylim(*y_lim)
    ax_median.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}"))
    ax_median.xaxis.set_major_locator(plt.MultipleLocator(5_000))   
    ax_median.grid(False)
    ax_median.tick_params(which='both', width=0.8, direction='in', bottom=True, top=True, left=True, right=True)
    ax_median.tick_params(which='major', length=3.5)
    ax_median.tick_params(which='minor', length=3.5)


def plot_individual_curves(
    cfg: dict,
    seed_results: list[dict],
    y_lim: tuple[float, float],
    results_dir: Path,
) -> None:
    """Plot and save mean/median curves for each seed into its logs dir."""

    plt.style.use("seaborn-v0_8-whitegrid")

    for result in seed_results:
        seed     = result["seed"]
        seed_dir = Path(result["seed_dir"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

        _plot_curve_panels(
            ax_mean=axes[0],
            ax_median=axes[1],
            mean_curve=result["mean_learning_curve"],
            median_curve=result["median_learning_curve"],
            auc_mean=result["auc_mean"],
            auc_median=result["auc_median"],
            y_lim=y_lim,
            title_suffix=f"Seed {seed}",
            score_max=cfg["evaluation"].get("score_max", float("inf")),
            score_min=cfg["evaluation"].get("score_min", -float("inf"))          
        )

        fig.suptitle(
            f"Learning Curves — {cfg['env_name']} · {cfg['method']} · Seed {seed}",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()

        path = seed_dir / f"learning_curve_seed_{seed}"
        _save_fig(fig, path)
        plt.close(fig)
        print(f"  Curve seed {seed}    →  {path}.png")


def plot_aggregated_curves(
    cfg: dict,
    agg_curves: dict,
    auc_agg_mean: float,
    auc_agg_median: float,
    y_lim: tuple[float, float],
    results_dir: Path,
) -> None:
    """Plot and save aggregated curves into results_dir."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    _plot_curve_panels(
        ax_mean=axes[0],
        ax_median=axes[1],
        mean_curve=agg_curves["aggregated_mean_curve"],
        median_curve=agg_curves["aggregated_median_curve"],
        auc_mean=auc_agg_mean,
        auc_median=auc_agg_median,
        y_lim=y_lim,
        title_suffix=f"{len(cfg["training"]["seeds"])} seeds aggregated",
    )

    fig.suptitle(
        f"Aggregated Learning Curves — {cfg["env_name"]} · {cfg["method"]}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = results_dir / f"learning_curve_aggregated"
    _save_fig(fig, path)
    plt.close(fig)
    print(f"  Curve aggregated    →  {path}.png")


# =============================================================================
# MAIN
# =============================================================================

def main(cfg: dict = None) -> None:
    """
    Run full training phase using best hyperparameters from tuning.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary injected by run_experiment.py.
        If None, loads CartPole-v1.yaml with RL_PURE as standalone fallback.

    Example usage
    -------------
    # Standalone
    main()

    # As module
    import train
    train.main(cfg)
    """

    if cfg is None:
        import yaml
        with open("configs/CartPole-v1.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["method"] = "RL_PURE"
        cfg["tuning"]["tuning_dir"]     = f"experiments/{cfg['env_name']}/RL_PURE/tuning"
        cfg["training"]["training_dir"] = f"experiments/{cfg['env_name']}/RL_PURE/training"

    env_name        = cfg["env_name"]
    method          = cfg["method"]
    seeds_training  = cfg["training"]["seeds"]
    training_dir    = Path(cfg["training"]["training_dir"])

    set_global_seeds(seeds_training[0])

    print(f"\nTraining directory: {training_dir}/\n")
    print(f"  env_name : {env_name}")
    print(f"  method   : {method}\n")

    # ── Load best hyperparameters ─────────────────────────────────────────────
    best_params, alpha = load_best_params(cfg)
    print(f"  Params: {best_params}\n")

    # ── Train across seeds ────────────────────────────────────────────────────
    print("Training:")
    seed_results = []
    start = time.perf_counter()


    for seed in seeds_training:
        result = train_seed(seed, best_params, alpha, cfg, training_dir)
        seed_results.append(result)

    elapsed = time.perf_counter() - start

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_curves     = build_aggregated_curves(seed_results)
    auc_agg_mean   = compute_auc(agg_curves["aggregated_mean_curve"],   value_key="mean")
    auc_agg_median = compute_auc(agg_curves["aggregated_median_curve"], value_key="median")
    y_lim          = compute_y_limits(seed_results)

    # ── Report ────────────────────────────────────────────────────────────────
    aucs_mean   = [r["auc_mean"]   for r in seed_results]
    aucs_median = [r["auc_median"] for r in seed_results]

    print(f"\n{'='*50}")
    print(f"  Environment:               {cfg["env_name"]}")
    print(f"  Method:                    {cfg["method"]}")
    print(f"  n_envs:                    {cfg["n_envs"]}")
    print(f"  Training total time:       {elapsed:.2f} sec")
    print(f"  Mean time per seed:        {elapsed / len(cfg["training"]["seeds"]):.2f} sec")
    print(f"  AUC (mean)   mean ± std:   {np.mean(aucs_mean):.2f} ± {np.std(aucs_mean):.2f}")
    print(f"  AUC IQM:                   {bootstrap_iqm_ci(aucs_median)}")
    print(f"{'='*50}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Saving outputs:")
    save_training_results(seed_results, agg_curves, training_dir)
    save_training_summary(cfg, seed_results, agg_curves, training_dir, elapsed)
    plot_individual_curves(cfg, seed_results, y_lim, training_dir)
    plot_aggregated_curves(cfg, agg_curves, auc_agg_mean, auc_agg_median, y_lim, training_dir)


if __name__ == "__main__":
    main()