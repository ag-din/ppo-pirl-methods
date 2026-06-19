# =============================================================================
# tuning.py — Hyperparameter tuning for PPO using Optuna
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
import optuna
import stable_baselines3
import torch
from optuna.importance import FanovaImportanceEvaluator
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# ── Internal ──────────────────────────────────────────────────────────────────
from src.env_utils import make_env, get_eval_method
from src.callbacks import PhysicsPolicyCallback
from src.physics import get_phi


# =============================================================================
# UTILS
# =============================================================================

def set_global_seeds(seed: int = 42) -> None:
    """
    Fix all sources of randomness for reproducibility.

    Parameters
    ----------
        seed : int
            The seed to use for all random number generators.
    
    Returns
    -------
        None
    """
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
# OPTUNA OBJECTIVE
# =============================================================================

def make_objective(cfg: dict):
    """
    Factory that returns an Optuna objective function closed over cfg.

    The returned objective trains a PPO agent under the candidate
    hyperparameter configuration for multiple random seeds and returns a
    robust performance metric for Optuna to optimize.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    objective : callable(trial: optuna.Trial) → float
        The Optuna objective function to optimize.
    """

    env_name = cfg["env_name"]
    method = cfg["method"]
    n_envs = cfg["n_envs"]
    timesteps_tuning = cfg["tuning"]["timesteps"]
    n_eval_episodes = cfg["tuning"]["n_eval_episodes"]
    seeds_tuning = cfg["tuning"]["seeds"]
    cfg_eval_seed = cfg["tuning"]["eval_seed"]

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective: train PPO across multiple seeds

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object representing the current hyperparameter
            configuration.
        
        Returns
        -------
        median_reward : float
            The median reward across seeds for the current hyperparameter
            configuration, used as the optimization metric.

        """

        # ── Sample hyperparameters ────────────────────────────────────────────
        # Candidate values for PPO hyperparameters.
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.90, 0.99, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
        n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20, 30])
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)

        # Physics prior weight — only tuned for reward-shaping and policy-based
        # PIRL methods. RL_PURE and PIRL_STATE do not use the α scaling here.
        alpha = (
            trial.suggest_float("alpha", 1e-6, 1.0, log=True)
            if method not in ["RL_PURE", "PIRL_STATE"]
            else 0.0
        )

        n_steps = trial.suggest_categorical("n_steps",
                [128, 256, 512, 1024, 2048])

        batch_size = trial.suggest_categorical("batch_size",
                [32, 64, 128, 256, 512])


        # ── Evaluate across seeds ─────────────────────────────────────────────
        rewards = []
        items   = []

        for seed in seeds_tuning:

            # ── Training env — with PIRL wrapper if applicable ────────────
            train_env = make_env(
                env_name=env_name,
                method=method,
                seed=seed,
                n_envs=n_envs,
                alpha=alpha,
                training=True,
            )

            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                n_epochs=n_epochs,
                ent_coef=ent_coef,
                clip_range=clip_range,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                verbose=1,
                seed=seed,
            )

            # ── Callback for PIRL_POLICY ──────────────────────────────────────
            callbacks = []
            if method == "PIRL_POLICY":
                phi_fn = get_phi(env_name)
                callbacks.append(PhysicsPolicyCallback(phi_fn=phi_fn, alpha=alpha))

            model.learn(timesteps_tuning, callback=callbacks or None)

            # ── Eval env — use the evaluation wrapper method chosen by get_eval_method.
            # This ensures the metric is computed consistently for the selected
            # PIRL variant, often comparing against the original unmodified reward.
            eval_seed = cfg_eval_seed
            eval_env = make_env(
                env_name=env_name,
                method=get_eval_method(method),
                seed=eval_seed,
                n_envs=n_envs,
                alpha=alpha,
                training=False,
            )

            # Carry over normalization statistics from training to evaluation.
            # This keeps VecNormalize behavior consistent between train and eval.
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms

            train_env.close()


            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            eval_env.close()

            rewards.append(float(mean_reward))
            items.append({
                "train_seed":  seed,
                "eval_seed":   eval_seed,
                "mean_reward": float(mean_reward),
                "std_reward":  float(std_reward),
            })

        # ── Aggregate ─────────────────────────────────────────────────────────
        mean_r   = float(np.mean(rewards))
        median_r = float(np.median(rewards))
        std_r    = float(np.std(rewards))

        trial.set_user_attr("mean_reward",   mean_r)
        trial.set_user_attr("median_reward", median_r)
        trial.set_user_attr("std_reward",    std_r)
        trial.set_user_attr("batch_size",    int(batch_size))
        trial.set_user_attr("alpha",         alpha)
        trial.set_user_attr("info_per_seed", items)

        return median_r

    return objective


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_tuning_results(study: optuna.Study, cfg: dict, results_dir: Path) -> Path:
    """
    Save all completed trials to a JSONL file.
    
    Parameters
    ----------
    study : optuna.Study
        The Optuna study containing all trials to save.
    cfg : dict
        Configuration dictionary.
    results_dir : Path
        Directory where the results file will be saved.

    Returns
    -------
    output_path : Path
        The path to the saved JSONL file containing trial results.
    """

    env_name = cfg["env_name"]
    method   = cfg["method"]

    output_path = results_dir / f"trials.jsonl"

    with output_path.open("w") as f:
        for t in study.trials:
            if t.value is None:
                continue

            record = {
                # Identification
                "trial":            t.number,
                "value":            t.value,

                # Experiment config
                "env_name":         env_name,
                "method":           method,
                "num_seeds":        len(cfg["tuning"]["seeds"]),
                "timesteps":        cfg["tuning"]["timesteps"],
                "n_eval_episodes":  cfg["tuning"]["n_eval_episodes"],
                "n_envs":           cfg["n_envs"],

                # Rewards
                "mean_reward":      t.user_attrs.get("mean_reward"),
                "median_reward":    t.user_attrs.get("median_reward"),
                "std_reward":       t.user_attrs.get("std_reward"),
                "info_per_seed":    t.user_attrs.get("info_per_seed"),

                # Hyperparameters
                "lr":               t.params.get("lr"),
                "n_steps":          t.params.get("n_steps"),
                "batch_size":       t.user_attrs.get("batch_size"),
                "gamma":            t.params.get("gamma"),
                "gae_lambda":       t.params.get("gae_lambda"),
                "n_epochs":         t.params.get("n_epochs"),
                "ent_coef":         t.params.get("ent_coef"),
                "clip_range":       t.params.get("clip_range"),
                "max_grad_norm":    t.params.get("max_grad_norm"),
                "vf_coef":         t.params.get("vf_coef"),
                "alpha":            t.user_attrs.get("alpha", 1.0),

                # Reproducibility
                "optuna_version":   optuna.__version__,
                "sb3_version":      stable_baselines3.__version__,
                "python_version":   platform.python_version(),
                "date":             datetime.now().isoformat(),
                "sampler":          type(study.sampler).__name__,
                "sampler_seed":     cfg["tuning"]["sampler_seed"],
                "pruner":           type(study.pruner).__name__,
                "direction":        study.direction.name,
            }

            f.write(json.dumps(record) + "\n")

    print(f"  Tuning results  →  {output_path}")
    return output_path


def save_best_params(study: optuna.Study, cfg: dict, results_dir: Path) -> Path:
    """
    Save best hyperparameters to JSON.
    
    Parameters
    ----------
    study : optuna.Study
        The Optuna study containing the best trial.
    cfg : dict
        Configuration dictionary.
    results_dir : Path
        Directory where the best parameters file will be saved.

    Returns
    -------
    output_path : Path
        The path to the saved JSON file containing the best hyperparameters.
    """

    env_name = cfg["env_name"]
    method = cfg["method"]

    best_params = study.best_params.copy()
    best_params["learning_rate"] = best_params.pop("lr")
    best_params["batch_size"] = study.best_trial.user_attrs.get("batch_size")
    best_params["alpha"] = study.best_trial.user_attrs.get("alpha", 1.0)
    if "batch_fraction" in best_params:
        del best_params["batch_fraction"]
    if "alpha" in best_params and "alpha" in study.best_params:
        pass

    output_path = results_dir / f"best_params.json"
    output_path.write_text(json.dumps(best_params, indent=2))
    print(f"  Best params     →  {output_path}")
    return output_path


def save_tuning_summary(study: optuna.Study, cfg: dict, results_dir: Path, elapsed: float) -> Path:
    """
    Save experiment-level summary to JSON.
    
    Parameters
    ----------
    study : optuna.Study
        The Optuna study containing all trials and the best trial.
    cfg : dict
        Configuration dictionary.
    results_dir : Path
        Directory where the summary file will be saved.
    elapsed : float
        Total time taken for the tuning phase in seconds.
        
    Returns
    -------
    output_path : Path
        The path to the saved JSON file containing the experiment summary.

    """

    env_name = cfg["env_name"]
    method   = cfg["method"]

    summary = {
        # Experiment
        "env_name":                  env_name,
        "method":                    method,
        "direction":                 cfg["tuning"]["direction"],

        # Compute
        "n_jobs":                    cfg["tuning"]["n_jobs"],
        "n_envs":                    cfg["n_envs"],
        "tuning_total_time_sec":     round(elapsed, 2),
        "mean_time_per_trial_sec":   round(elapsed / cfg["tuning"]["n_trials"], 2),

        # Trials
        "n_trials_requested":        cfg["tuning"]["n_trials"],
        "n_trials_completed":        len([t for t in study.trials if t.value is not None]),
        "n_trials_failed":           len([t for t in study.trials if t.value is None]),

        # Best
        "best_trial":                study.best_trial.number,
        "best_value":                round(study.best_value, 4),

        # Reproducibility
        "sampler_seed":              cfg["tuning"]["sampler_seed"],
        "seeds_tuning":              cfg["tuning"]["seeds"],
        "timesteps_tuning":          cfg["tuning"]["timesteps"],
        "n_eval_episodes_tuning":    cfg["tuning"]["n_eval_episodes"],
        "date":                      datetime.now().isoformat(),
        "optuna_version":            optuna.__version__,
        "sb3_version":               stable_baselines3.__version__,
        "python_version":            platform.python_version(),
    }

    output_path = results_dir / f"summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary         →  {output_path}")
    return output_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_param_importances(study: optuna.Study, cfg: dict, results_dir: Path) -> None:
    """
    Plot and save hyperparameter importances using fANOVA.
    
    Parameters
    ----------
    study : optuna.Study
        The Optuna study containing all trials to analyze for importance.
    cfg : dict
        Configuration dictionary.
    results_dir : Path
        Directory where the importance plot will be saved.  

    Returns
    -------
    None
    """

    env_name = cfg["env_name"]
    method   = cfg["method"]

    importances = optuna.importance.get_param_importances(
        study,
        evaluator=FanovaImportanceEvaluator(),
    )

    rename = {
        "lr": "learning_rate",
    }

    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=False)
    labels = [rename.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    if method == "RL_PURE":
        labels.insert(0, "alpha")
        values.insert(0, 0.0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(labels, values)
    ax.set_xlabel("Importance (fANOVA)", fontsize=11)
    ax.set_ylabel("Hyperparameter", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Hyperparameter Importance (fANOVA)\n{env_name} · {method}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = results_dir / f"param_importance.png"
    plt.savefig(output_path, format="png", bbox_inches="tight")
    print(f"  Importance plot →  {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main(cfg: dict = None) -> None:
    """
    Run hyperparameter tuning for PPO.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Example usage
    -------------
    # Run with a provided config dictionary
    import tuning
    tuning.main(cfg)

    Notes
    -----
    This module expects a configuration dictionary to be passed in. The
    current entry point does not load a default config from disk.

    Returns
    -------
    None
    """
    effective_cfg = cfg

    env_name        = effective_cfg["env_name"]
    method          = effective_cfg["method"]
    n_trials        = effective_cfg["tuning"]["n_trials"]
    sampler_seed    = effective_cfg["tuning"]["sampler_seed"]
    direction       = effective_cfg["tuning"]["direction"]
    n_jobs          = effective_cfg["tuning"]["n_jobs"]
    results_dir     = Path(effective_cfg["tuning"]["tuning_dir"])

    set_global_seeds(sampler_seed)

    # ── Output directory ──────────────────────────────────────────────────────
    print(f"\nResults directory: {results_dir}/\n")
    print(f"  env_name : {env_name}")
    print(f"  method   : {method}\n")

    # ── Create study ──────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=f"{env_name}_{method}",
        storage=f"sqlite:///experiments/{env_name}/{method}/tuning/tuning.db",
        load_if_exists=False,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.NopPruner(),
    )

    # ── Optimize ──────────────────────────────────────────────────────────────
    start = time.perf_counter()
    study.optimize(
        make_objective(effective_cfg),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=n_jobs,
    )
    elapsed = time.perf_counter() - start

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Saving outputs:")
    save_tuning_results(study, effective_cfg, results_dir)
    save_best_params(study, effective_cfg, results_dir)
    save_tuning_summary(study, effective_cfg, results_dir, elapsed)
    plot_param_importances(study, effective_cfg, results_dir)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  n_jobs:               {n_jobs}")
    print(f"  n_envs:               {effective_cfg['n_envs']}")
    print(f"  Tuning total time:    {elapsed:.2f} sec")
    print(f"  Mean time per trial:  {elapsed / n_trials:.2f} sec")
    print(f"  Best value:           {study.best_value:.4f}")
    print(f"  Best params:          {study.best_params}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()