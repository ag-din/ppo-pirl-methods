# =============================================================================
# evaluate.py — Final evaluation of trained models using rliable metrics
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
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Internal ──────────────────────────────────────────────────────────────────
from src.env_utils import make_env, get_eval_method

# rliable
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


# =============================================================================
# UTILS
# =============================================================================

def get_plots_dir(eval_dir: Path) -> Path:
    """Return and create the plots subdirectory."""
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def get_data_dir(eval_dir: Path) -> Path:
    """Return and create the data subdirectory."""
    data_dir = eval_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_seed_model_path(cfg: dict, seed: int) -> Path:
    """Return path to best_model.zip for a given seed from training output."""
    training_dir = Path(cfg["training"]["training_dir"])
    return training_dir / f"logs_seed_{seed}" / "best_model.zip"


def get_seed_vecnorm_path(cfg: dict, seed: int) -> Path:
    """Return path to vecnormalize.pkl for a given seed from training output."""
    training_dir = Path(cfg["training"]["training_dir"])
    return training_dir / f"logs_seed_{seed}" / "vecnormalize.pkl"


def set_global_seeds(seed: int = 42) -> None:
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure as PNG only."""
    fig.savefig(path.with_suffix(".png"), format="png", bbox_inches="tight")


def _normalize(rewards: list[float], score_min: float, score_max: float) -> list[float]:
    """Normalize rewards to [0, 1] using score_min and score_max."""
    return [(r - score_min) / (score_max - score_min) for r in rewards]


def _scalar(x: np.ndarray) -> float:
    """Safely extract a scalar from any numpy array shape."""
    return float(np.asarray(x).flat[0])


def _denormalize(val: float, score_min: float, score_max: float) -> float:
    """Convert normalized score [0,1] back to original reward scale."""
    return val * (score_max - score_min) + score_min


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(cfg: dict, seed: int) -> PPO:
    """Load best_model.zip from training output for a given seed."""

    model_path = get_seed_model_path(cfg, seed)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run training phase first."
        )

    model = PPO.load(str(model_path))
    print(f"  Loaded model seed {seed} ← {model_path}")
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_seed(seed: int, cfg: dict, data_dir: Path) -> dict:
    """Run n_eval_episodes for one seed. Returns per-seed results."""

    env_name        = cfg["env_name"]
    method          = cfg["method"]
    n_eval_episodes = cfg["evaluation"]["n_eval_episodes"]
    eval_seed       = cfg["evaluation"]["eval_seed"]
    deterministic   = cfg["evaluation"]["deterministic"]

    model       = load_model(cfg, seed)
    vecnorm_path = get_seed_vecnorm_path(cfg, seed)

    # Load the correct eval env with saved VecNormalize stats from training
    eval_env = make_env(
        env_name=env_name,
        method=get_eval_method(method),
        seed=eval_seed,
        n_envs=1,
        alpha=cfg.get("alpha", 0.0),
        vecnorm_path=str(vecnorm_path) if vecnorm_path.exists() else None,
        training=False,
    )

    episode_rewards = []
    for _ in range(n_eval_episodes):
        r, _ = evaluate_policy(
            model, eval_env,
            n_eval_episodes=1,
            deterministic=deterministic,
        )
        episode_rewards.append(float(r))

    eval_env.close()

    arr = np.array(episode_rewards)

    record = {
        "seed":             seed,
        "n_eval_episodes":  n_eval_episodes,
        "deterministic":    deterministic,
        "eval_seed":        eval_seed,
        "mean":             round(float(np.mean(arr)), 8),
        "median":           round(float(np.median(arr)), 8),
        "std":              round(float(np.std(arr)), 8),
        "iqr":              round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 8),
        "min":              round(float(np.min(arr)), 8),
        "max":              round(float(np.max(arr)), 8),
        "episode_rewards":  [round(r, 8) for r in episode_rewards],

        # Reproducibility
        "model_path":       str(get_seed_model_path(cfg, seed)),
        "env_name":         env_name,
        "method":           method,
        "date":             datetime.now().isoformat(),
        "sb3_version":      stable_baselines3.__version__,
        "python_version":   platform.python_version(),
    }

    # Save per-seed JSONL inside data/
    output_path = data_dir / f"seed_{seed}.jsonl"
    with output_path.open("w") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"  Seed {seed}: mean={record['mean']:.2f}  "
        f"median={record['median']:.2f}  "
        f"std={record['std']:.2f}  "
        f"→ {output_path}"
    )

    return record


# =============================================================================
# RLIABLE METRICS
# =============================================================================

def compute_rliable_metrics(seed_records: list[dict], cfg: dict, data_dir: Path) -> dict:
    """Compute IQM, median, mean + 95% CIs via rliable."""

    method          = cfg["method"]
    score_min       = cfg["evaluation"]["score_min"]
    score_max       = cfg["evaluation"]["score_max"]
    n_bootstrap     = cfg["evaluation"]["n_bootstrap"]
    n_eval_episodes = cfg["evaluation"]["n_eval_episodes"]

    all_rewards  = np.array([r["episode_rewards"] for r in seed_records])  # (n_seeds, n_eval)
    scores_norm  = (all_rewards - score_min) / (score_max - score_min)

    # rliable format: {method: (n_runs, n_eval, n_games)} → n_games=1
    score_dict = {method: scores_norm[:, :, np.newaxis]}

    # ── Each metric called separately ─────────────────────────────────────────
    iqm_scores,    iqm_cis    = rly.get_interval_estimates(
        score_dict, metrics.aggregate_iqm,    reps=n_bootstrap)

    mean_scores,   mean_cis   = rly.get_interval_estimates(
        score_dict, metrics.aggregate_mean,   reps=n_bootstrap)

    median_scores, median_cis = rly.get_interval_estimates(
        score_dict, metrics.aggregate_median, reps=n_bootstrap)

    # ── Raw metrics across seeds ──────────────────────────────────────────────
    seed_means = np.array([r["mean"] for r in seed_records])

    rliable_summary = {
        "env_name":             cfg["env_name"],
        "method":               method,
        "n_seeds":              len(seed_records),
        "n_eval_episodes":      n_eval_episodes,
        "score_min":            score_min,
        "score_max":            score_max,

        # Raw
        "mean_of_seed_means":   round(float(np.mean(seed_means)), 8),
        "std_of_seed_means":    round(float(np.std(seed_means)), 8),
        "median_of_seed_means": round(float(np.median(seed_means)), 8),
        "iqr_of_seed_means":    round(float(
            np.percentile(seed_means, 75) - np.percentile(seed_means, 25)
        ), 8),

        # rliable normalized
        "iqm":            round(_scalar(iqm_scores[method]),      8),
        "iqm_ci_low":     round(_scalar(iqm_cis[method][0]),      8),
        "iqm_ci_high":    round(_scalar(iqm_cis[method][1]),      8),
        "mean_norm":      round(_scalar(mean_scores[method]),     8),
        "mean_ci_low":    round(_scalar(mean_cis[method][0]),     8),
        "mean_ci_high":   round(_scalar(mean_cis[method][1]),     8),
        "median_norm":    round(_scalar(median_scores[method]),   8),
        "median_ci_low":  round(_scalar(median_cis[method][0]),   8),
        "median_ci_high": round(_scalar(median_cis[method][1]),   8),

        # rliable denormalized (original reward scale)
        "iqm_raw":            round(_denormalize(_scalar(iqm_scores[method]),    score_min, score_max), 8),
        "iqm_raw_ci_low":     round(_denormalize(_scalar(iqm_cis[method][0]),    score_min, score_max), 8),
        "iqm_raw_ci_high":    round(_denormalize(_scalar(iqm_cis[method][1]),    score_min, score_max), 8),
        "mean_raw":           round(_denormalize(_scalar(mean_scores[method]),   score_min, score_max), 8),
        "mean_raw_ci_low":    round(_denormalize(_scalar(mean_cis[method][0]),   score_min, score_max), 8),
        "mean_raw_ci_high":   round(_denormalize(_scalar(mean_cis[method][1]),   score_min, score_max), 8),
        "median_raw":         round(_denormalize(_scalar(median_scores[method]), score_min, score_max), 8),
        "median_raw_ci_low":  round(_denormalize(_scalar(median_cis[method][0]), score_min, score_max), 8),
        "median_raw_ci_high": round(_denormalize(_scalar(median_cis[method][1]), score_min, score_max), 8),

        "n_bootstrap_samples":  n_bootstrap,
        "date":                 datetime.now().isoformat(),
    }

    output_path = data_dir / "rliable_summary.json"
    output_path.write_text(json.dumps(rliable_summary, indent=2))
    print(f"  rliable summary     →  {output_path}")

    return {
        "summary":        rliable_summary,
        "score_dict":     score_dict,
        "iqm_scores":     iqm_scores,
        "iqm_cis":        iqm_cis,
        "mean_scores":    mean_scores,
        "mean_cis":       mean_cis,
        "median_scores":  median_scores,
        "median_cis":     median_cis,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def _plot_histogram(
    ax: plt.Axes,
    rewards: list[float],
    title: str,
    density: bool = False,
) -> None:
    """Plot a single histogram panel with mean and median lines."""
    mean   = float(np.mean(rewards))
    median = float(np.median(rewards))
    ylabel = "Density" if density else "Count"

    ax.hist(rewards, bins=20, edgecolor="white", alpha=0.85, density=density)
    ax.axvline(mean,   color="red",    linestyle="--", linewidth=1.5, label=f"Mean={mean:.1f}")
    ax.axvline(median, color="orange", linestyle="--", linewidth=1.5, label=f"Median={median:.1f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Episode Return")
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.legend(fontsize=10)
    ax.tick_params(which='both', width=0.8, direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params(which='major', length=3.5)
    ax.tick_params(which='minor', length=3.5)


def plot_seed_distributions(seed_records: list[dict], cfg: dict, plots_dir: Path) -> None:
    """Plot per-seed distributions — one PNG per seed."""

    env_name = cfg["env_name"]
    method   = cfg["method"]

    plt.style.use("seaborn-v0_8-whitegrid")

    for record in seed_records:
        seed    = record["seed"]
        rewards = record["episode_rewards"]

        fig, axes = plt.subplots(2, 1, figsize=(7, 8))

        _plot_histogram(axes[0], rewards, title="Count",   density=False)
        _plot_histogram(axes[1], rewards, title="Density", density=True)

        fig.suptitle(
            f"Return Distribution — {env_name} · {method} · Seed {seed}",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        path = plots_dir / f"distribution_seed_{seed}"
        _save_fig(fig, path)
        #plt.show()
        plt.close(fig)
        print(f"  Seed {seed} distribution  →  {path}.png")


def plot_aggregated_distribution(seed_records: list[dict], cfg: dict, plots_dir: Path) -> None:
    """Plot aggregated distribution across all seeds."""

    env_name = cfg["env_name"]
    method   = cfg["method"]

    plt.style.use("seaborn-v0_8-whitegrid")
    all_rewards = [r for record in seed_records for r in record["episode_rewards"]]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    _plot_histogram(axes[0], all_rewards, title="Count",   density=False)
    _plot_histogram(axes[1], all_rewards, title="Density", density=True)

    fig.suptitle(
        f"Aggregated Return Distribution — {env_name} · {method}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = plots_dir / "distribution_aggregated"
    _save_fig(fig, path)
    #plt.show()
    plt.close(fig)
    print(f"  Aggregated distribution  →  {path}.png")


def plot_performance_profile(
    rliable_results: dict,
    cfg: dict,
    plots_dir: Path,
    data_dir: Path | None = None,          # ← nuevo parámetro opcional
) -> None:
    """Plot performance profile (CDF of scores) using rliable.
    
    If data_dir is provided, saves the profile data as JSON alongside the plot.
    """

    method      = cfg["method"]
    n_bootstrap = cfg["evaluation"]["n_bootstrap"]
    score_dict  = rliable_results["score_dict"]
    thresholds  = np.linspace(0.0, 1.0, 100)

    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict,
        thresholds,
        reps=n_bootstrap,
    )

    # ── Save profile data to JSON ─────────────────────────────────────────────
    if data_dir is not None:
        profile_data = {
            "env_name":                   cfg["env_name"],
            "method":                     method,
            "n_bootstrap":                n_bootstrap,
            "thresholds":                 thresholds.tolist(),
            "score_distribution":         score_distributions[method].tolist(),
            "score_distribution_ci_low":  score_distributions_cis[method][0].tolist(),
            "score_distribution_ci_high": score_distributions_cis[method][1].tolist(),
            "date":                       datetime.now().isoformat(),
        }
        profile_path = data_dir / "performance_profile.json"
        profile_path.write_text(json.dumps(profile_data, indent=2))
        print(f"  Performance profile data →  {profile_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_utils.plot_performance_profiles(
        score_distributions,
        thresholds,
        performance_profile_cis=score_distributions_cis,
        colors={method: "steelblue"},
        ax=ax,
    )

    ax.set_xlabel("Normalized Score τ", fontsize=11)
    ax.set_ylabel("Fraction of runs with score > τ", fontsize=11)
    ax.set_title(
        f"Performance Profile — {cfg['env_name']} · {method}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = plots_dir / "performance_profile"
    _save_fig(fig, path)
    plt.close(fig)
    print(f"  Performance profile  →  {path}.png")


def plot_iqm_with_ci(rliable_results: dict, cfg: dict, plots_dir: Path) -> None:
    """Plot IQM, mean and median with 95% bootstrapped CIs."""

    method = cfg["method"]

    metrics_data = [
        ("IQM",    rliable_results["iqm_scores"],    rliable_results["iqm_cis"]),
        ("Mean",   rliable_results["mean_scores"],   rliable_results["mean_cis"]),
        ("Median", rliable_results["median_scores"], rliable_results["median_cis"]),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    for ax, (label, scores, cis) in zip(axes, metrics_data):
        val = _scalar(scores[method])
        lo  = _scalar(cis[method][0])
        hi  = _scalar(cis[method][1])

        ax.barh([0], [hi - lo], left=[lo], height=0.4,
                color="steelblue", alpha=0.4, label="95% CI")
        ax.scatter([val], [0], color="steelblue", zorder=5, s=80, label=label)
        ax.annotate(f"{val:.4f}  [{lo:.4f}, {hi:.4f}]",
                    xy=(val, 0), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10)

        ax.set_ylabel(label, fontsize=11, rotation=0, labelpad=55, va="center")
        ax.set_yticks([])
        ax.set_xlim(max(0, lo - 0.05), min(1, hi + 0.05))
        ax.set_xlabel("Normalized Score" if ax == axes[-1] else "")

    fig.suptitle(
        f"Metrics + 95% CI — {cfg['env_name']} · {method}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = plots_dir / "metrics_ci"
    _save_fig(fig, path)
    #plt.show()
    plt.close(fig)
    print(f"  Metrics + CI plot   →  {path}.png")


# =============================================================================
# VIDEO RECORDING
# =============================================================================

def record_videos(cfg: dict, eval_dir: Path) -> None:
    """
    Record n_video_episodes using the trained model with correct VecNormalize.
 
    Uses VecNormalize with saved training stats so the model receives
    normalized observations — identical to the evaluation environment.
    Frames captured manually from the base env and saved as MP4 via imageio.
    """
    try:
        import imageio
    except ImportError:
        print("  [SKIP] imageio not installed. Run: pip install imageio[ffmpeg]")
        return
 
    env_name         = cfg["env_name"]
    method           = cfg["method"]
    n_video_episodes = cfg["evaluation"]["n_video_episodes"]
    eval_seed        = cfg["evaluation"]["eval_seed"]
    deterministic    = cfg["evaluation"]["deterministic"]
    video_seed       = cfg["training"]["seeds"][0]
 
    model        = load_model(cfg, video_seed)
    vecnorm_path = get_seed_vecnorm_path(cfg, video_seed)
    video_dir    = eval_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
 
    # ── Build env with render_mode + VecNormalize ─────────────────────────────
    from src.env_utils import _WRAPPED_METHODS
    from src.physics import get_phi
    from src.wrappers import make_wrapper
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
 
    eval_method = get_eval_method(method)
    phi         = get_phi(env_name) if eval_method in _WRAPPED_METHODS else None
    alpha       = cfg.get("alpha", 0.0)
 
    def _init_render():
        env = gym.make(env_name, render_mode="rgb_array")
        if eval_method in _WRAPPED_METHODS:
            env = make_wrapper(env, eval_method, phi, alpha)
        return env
 
    venv = make_vec_env(_init_render, n_envs=1, seed=eval_seed,
                        vec_env_cls=DummyVecEnv)
 
    # Load saved VecNormalize stats from training
    if vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training    = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
        print("  [WARN] vecnormalize.pkl not found — using fresh stats")
 
    # ── Capture frames ────────────────────────────────────────────────────────
    episode       = 0
    episode_frames = []
    obs            = venv.reset()

    while episode < n_video_episodes:
        frame = venv.envs[0].render()
        if frame is not None:
            episode_frames.append(frame)

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = venv.step(action)

        if done[0]:
            # Save this episode's video
            if episode_frames:
                video_path = video_dir / f"{env_name}_{method}_seed_{video_seed}_ep{episode+1}.mp4"
                imageio.mimsave(str(video_path), episode_frames, fps=30, macro_block_size=1)
                print(f"  Episode {episode+1}/{n_video_episodes} → {video_path.name}")

            episode       += 1
            episode_frames = []  # reset for next episode

            if episode < n_video_episodes:
                obs = venv.reset()

    venv.close()
    print(f"  Videos saved → {video_dir}/")



# =============================================================================
# MAIN
# =============================================================================

def main(cfg: dict = None) -> None:
    """
    Run full evaluation phase using trained models.

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
    import evaluate
    evaluate.main(cfg)
    """

    if cfg is None:
        import yaml
        with open("configs/CartPole-v1.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["method"] = "RL_PURE"
        cfg["training"]["training_dir"]     = f"experiments/{cfg['env_name']}/RL_PURE/training"
        cfg["evaluation"]["evaluation_dir"] = f"experiments/{cfg['env_name']}/RL_PURE/evaluation"

    env_name    = cfg["env_name"]
    method      = cfg["method"]
    seeds_eval  = cfg["training"]["seeds"]
    eval_dir    = Path(cfg["evaluation"]["evaluation_dir"])

    set_global_seeds(cfg["evaluation"]["eval_seed"])

    # ── Output directories ────────────────────────────────────────────────────
    data_dir  = get_data_dir(eval_dir)
    plots_dir = get_plots_dir(eval_dir)
    print(f"\nEvaluation directory: {eval_dir}/\n")
    print(f"  env_name : {env_name}")
    print(f"  method   : {method}\n")

    # ── Evaluate each seed ────────────────────────────────────────────────────
    print("Evaluating seeds:")
    seed_records = []
    start = time.perf_counter()

    for seed in seeds_eval:
        record = evaluate_seed(seed, cfg, data_dir)
        seed_records.append(record)

    elapsed = time.perf_counter() - start

    # ── rliable metrics ───────────────────────────────────────────────────────
    print("\nComputing rliable metrics:")
    rliable_results = compute_rliable_metrics(seed_records, cfg, data_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    s = rliable_results["summary"]
    print(f"\n{'='*50}")
    print(f"  Environment:               {env_name}")
    print(f"  Method:                    {method}")
    print(f"  Seeds:                     {seeds_eval}")
    print(f"  Episodes per seed:         {cfg['evaluation']['n_eval_episodes']}")
    print(f"─ Raw ──────────────────────────────────────────")
    print(f"  Mean  of seed means:       {s['mean_of_seed_means']:.2f}")
    print(f"  Std   of seed means:       {s['std_of_seed_means']:.2f}")
    print(f"  Median of seed means:      {s['median_of_seed_means']:.2f}")
    print(f"  IQR   of seed means:       {s['iqr_of_seed_means']:.2f}")
    print(f"─ rliable (normalized) ─────────────────────────")
    print(f"  IQM:      {s['iqm']:.4f}  [{s['iqm_ci_low']:.4f}, {s['iqm_ci_high']:.4f}]")
    print(f"  Mean:     {s['mean_norm']:.4f}  [{s['mean_ci_low']:.4f}, {s['mean_ci_high']:.4f}]")
    print(f"  Median:   {s['median_norm']:.4f}  [{s['median_ci_low']:.4f}, {s['median_ci_high']:.4f}]")
    print(f"─ rliable (original scale) ─────────────────────")
    print(f"  IQM:      {s['iqm_raw']:.2f}  [{s['iqm_raw_ci_low']:.2f}, {s['iqm_raw_ci_high']:.2f}]")
    print(f"  Mean:     {s['mean_raw']:.2f}  [{s['mean_raw_ci_low']:.2f}, {s['mean_raw_ci_high']:.2f}]")
    print(f"  Median:   {s['median_raw']:.2f}  [{s['median_raw_ci_low']:.2f}, {s['median_raw_ci_high']:.2f}]")
    print(f"─ Compute ──────────────────────────────────────")
    print(f"  Eval total time:           {elapsed:.2f} sec")
    print(f"{'='*50}\n")


    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting:")
    plot_seed_distributions(seed_records, cfg, plots_dir)
    plot_aggregated_distribution(seed_records, cfg, plots_dir)
    plot_performance_profile(rliable_results, cfg, plots_dir, data_dir=data_dir)
    plot_iqm_with_ci(rliable_results, cfg, plots_dir)

    # ── Videos ───────────────────────────────────────────────────────────────
    print("\nRecording videos:")
    record_videos(cfg, eval_dir)


if __name__ == "__main__":
    main()