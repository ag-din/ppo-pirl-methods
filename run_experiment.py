# =============================================================================
# run_experiment.py — Orchestrate tuning, training and evaluation
# =============================================================================
#
# Usage:
#   python run_experiment.py --config configs/CartPole-v1.yaml --method RL_PURE --only tuning
#   python run_experiment.py --config configs/CartPole-v1.yaml --method PIRL_REWARD
#   python run_experiment.py --config configs/CartPole-v1.yaml --method RL_PURE --skip tuning
#   python run_experiment.py --config configs/CartPole-v1.yaml --method RL_PURE --only evaluation --experiments-dir /path/to/results
#
# =============================================================================

# ── Libraries ──────────────────────────────────────────────────────────
import argparse
import time
import yaml
from datetime import datetime
from pathlib import Path

# =============================================================================
# ARG PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment pipeline."""

    parser = argparse.ArgumentParser(description="Run RL and PIRL experiment pipeline.")

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/CartPole-v1.yaml)"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["RL_PURE", "PIRL_REWARD", "PIRL_STATE", "PIRL_POLICY"],
        help="Method to run"
    )
    parser.add_argument(
        "--skip", type=str, nargs="+", default=[],
        choices=["tuning", "training", "evaluation"],
        help="Phases to skip (e.g. --skip tuning)"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["tuning", "training", "evaluation"],
        help="Run only this phase"
    )

    parser.add_argument(
        "--experiments-dir", type=str, default="experiments",
        help="Base directory for all experiment outputs (default: experiments/)"
    )

    return parser.parse_args()


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(config_path: str, method: str) -> dict:
    """
    Load the YAML experiment config and validate the selected method.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    method : str
        The method to run (e.g. RL_PURE, PIRL_REWARD, etc.). Must be declared
        in the config's "methods_available" section.  

    Returns
    -------
    cfg : dict
        The loaded and validated configuration dictionary, with the selected method
        stored under the "method" key for later use.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Validate that the requested method is declared in the config.
    if method not in cfg.get("methods_available", {}):
        raise ValueError(
            f"Method '{method}' not found in config wrappers. "
            f"Available: {cfg['methods_available']}"
        )

    # Store the chosen method on the loaded config for later phases.
    cfg["method"] = method

    return cfg


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
 
def setup_output_dirs(cfg: dict, experiments_dir: str) -> dict:
    """
    Create the experiment output tree and add output paths to cfg.
    
    Parameters
    ----------
    cfg : dict
        The loaded configuration dictionary.
    experiments_dir : str
        Base directory for all experiment outputs (e.g. "experiments/"). The
        directory structure will be: experiments/{env_name}/{method}/{phase}/

    Returns
    -------
    cfg : dict
        The input config with added output paths for each phase.
    """

    env_name = cfg["env_name"]
    method = cfg["method"]

    root = Path(experiments_dir) / env_name / method
    tuning_dir = root / "tuning"
    training_dir = root / "training"
    eval_dir = root / "evaluation"

    # Ensure all workspace directories exist before running phase code.
    for d in [tuning_dir, training_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Store output locations in the config for later use by each phase.
    cfg["output_dir"] = str(root)
    cfg["tuning"]["tuning_dir"] = str(tuning_dir)
    cfg["training"]["training_dir"] = str(training_dir)
    cfg["evaluation"]["evaluation_dir"] = str(eval_dir)

    return cfg


# =============================================================================
# PHASE RUNNERS
# =============================================================================

def run_tuning(cfg: dict) -> None:
    """
    Run hyperparameter tuning phase.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    None
    
    """

    import src.tuning as tuning_module

    tuning_module.main(cfg)


def run_training(cfg: dict) -> None:
    """
    Run training phase.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    None
    
    """

    import src.train as train_module

    train_module.main(cfg)


def run_evaluation(cfg: dict) -> None:
    """
    Run evaluation phase.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    None
    
    """

    import src.evaluate as evaluate_module

    evaluate_module.main(cfg)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the configured experiment phases in order and report timing."""

    args = parse_args()
    cfg = load_config(args.config, args.method)

    # ── Setup output directories ──────────────────────────────────────────────
    cfg = setup_output_dirs(cfg, args.experiments_dir)

    # ── Determine phases to run ───────────────────────────────────────────────
    all_phases = ["tuning", "training", "evaluation"]

    if args.only:
        # The --only option takes precedence over --skip: run exactly one phase.
        phases = [args.only]
    else:
        phases = [p for p in all_phases if p not in args.skip]

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Experiment:  {cfg['env_name']} · {cfg['method']}")
    print(f"  Config:      {args.config}")
    print(f"  Output:      {cfg['output_dir']}")
    print(f"  Phases:      {' → '.join(phases)}")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    total_start = time.perf_counter()

    # ── Run phases ────────────────────────────────────────────────────────────
    phase_runners = {
        "tuning":     run_tuning,
        "training":   run_training,
        "evaluation": run_evaluation,
    }

    for phase in phases:
        print(f"\n{'─'*55}")
        print(f"  ▶  Phase: {phase.upper()}")
        print(f"{'─'*55}\n")

        phase_start = time.perf_counter()
        phase_runners[phase](cfg)
        phase_elapsed = time.perf_counter() - phase_start

        print(f"\n  ✓  {phase.capitalize()} done in {phase_elapsed:.2f} sec")

    # ── Footer ────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*55}")
    print(f"  Experiment complete")
    print(f"  Total time: {total_elapsed:.2f} sec")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()