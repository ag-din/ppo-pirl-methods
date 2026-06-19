# =============================================================================
# run_comparison.py — Run all per-environment analyses sequentially
# =============================================================================
#
# Runs the three analysis scripts in src/ for a single environment, in order:
#   1. compare_final_performance.py   
#   2. compare_sample_efficiency.py   
#   3. compare_stability.py           
#
# Each script is invoked as a subprocess with the same --env and
# --experiments-dir arguments. If one fails, the orchestrator reports it
# and (by default) continues with the next; use --stop-on-error to abort.
#
# Usage:
#   python run_comparison.py --env Reacher-v5
#   python run_comparison.py --env Reacher-v5 --experiments-dir experiments
#   python run_comparison.py --env CartPole-v1 --experiments-dir experiments_results --stop-on-error
#
# =============================================================================

import argparse
import subprocess
import sys
from pathlib import Path


# Analysis scripts to run, in order. Paths are relative to src/.
ANALYSIS_SCRIPTS = [
    ("Final Performance", "compare_final_performance.py"),
    ("Sample Efficiency", "compare_sample_efficiency.py"),
    ("Cross-Seed Stability", "compare_stability.py"),
]

# Directory containing the analysis scripts (this file lives next to src/).
SRC_DIR = Path(__file__).resolve().parent / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all per-environment analyses sequentially."
    )
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name (e.g. Reacher-v5)")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Root experiments directory (default: experiments/)")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort if any analysis fails (default: continue)")
    return parser.parse_args()


def run_script(
    label:       str,
    script_name: str,
    env:         str,
    exp_dir:     str,
) -> bool:
    """Run one analysis script as a subprocess. Returns True on success."""
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        print(f"  [ERROR] Script not found: {script_path}")
        return False

    cmd = [
        sys.executable, str(script_path),
        "--env", env,
        "--experiments-dir", exp_dir,
    ]

    print(f"\n{'#'*70}")
    print(f"#  {label}")
    print(f"#  {' '.join(cmd)}")
    print(f"{'#'*70}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAIL] {label} exited with code {result.returncode}")
        return False

    print(f"  [OK]   {label} completed")
    return True


def main() -> None:
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"  Running analyses for environment: {args.env}")
    print(f"  Experiments directory: {args.experiments_dir}")
    print(f"{'='*70}")

    succeeded = []
    failed    = []

    for label, script_name in ANALYSIS_SCRIPTS:
        ok = run_script(label, script_name, args.env, args.experiments_dir)
        (succeeded if ok else failed).append(label)

        if not ok and args.stop_on_error:
            print(f"\n  [ABORT] Stopping because --stop-on-error is set.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Analysis summary for {args.env}")
    print(f"{'='*70}")
    for label in succeeded:
        print(f"  ✓ {label}")
    for label in failed:
        print(f"  ✗ {label}")
    print(f"{'='*70}\n")

    # Non-zero exit code if anything failed, so callers/CI can detect it.
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()