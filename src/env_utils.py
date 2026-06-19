# =============================================================================
# env_utils.py — Environment creation utilities for the PIRL benchmark
# =============================================================================
#
# Used by tuning.py, train.py and evaluate.py.
#
# All environments are created with VecNormalize applied regardless of domain.
# VecNormalize normalizes observations (mean=0, std=1) and rewards using
# running statistics updated during training. In evaluation, saved statistics
# are loaded with training=False so the normalization is fixed.
#
# Wrapper application rules:
#   RL_PURE      → no PIRL wrapper
#   PIRL_REWARD  → RewardShapingWrapper
#   PIRL_STATE   → StateAugmentationWrapper  (applied BEFORE VecNormalize)
#   PIRL_POLICY  → no PIRL wrapper (physics handled via SB3 callback)
#
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv

# ── Internal ──────────────────────────────────────────────────────────────────
from src.physics import get_phi
from src.wrappers import make_wrapper


# =============================================================================
# CONSTANTS
# =============================================================================

_WRAPPED_METHODS = {"PIRL_REWARD", "PIRL_STATE"}

# Evaluation method map — determines which wrapper (if any) the eval env needs.
# PIRL_REWARD: reward was only a training signal → eval with original reward (RL_PURE)
# PIRL_STATE:  model learned with [obs, φ] as input → eval needs same augmented obs
# PIRL_POLICY: only the loss changed, env interface unchanged → eval identical to RL_PURE
_EVAL_METHOD_MAP = {
    "RL_PURE":      "RL_PURE",
    "PIRL_REWARD":  "RL_PURE",
    "PIRL_STATE":   "PIRL_STATE",
    "PIRL_POLICY":  "RL_PURE",
}


def get_eval_method(method: str) -> str:
    """
    Return the method to use when creating the evaluation environment.

    The evaluation environment must match the interface the model was
    trained with — not necessarily the training method itself:

    - PIRL_REWARD and PIRL_POLICY do not modify the environment interface
      (only the reward signal or the loss). The model sees normal obs and
      actions, so evaluation uses RL_PURE to measure the original reward.

    - PIRL_STATE augments the observation with phi — the model expects
      [obs, phi] as input. Evaluation must use the same wrapper.

    Parameters
    ----------
    method : str — training method

    Returns
    -------
    str — method to use for the evaluation environment
    """
    if method not in _EVAL_METHOD_MAP:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Expected one of: {list(_EVAL_METHOD_MAP.keys())}"
        )
    return _EVAL_METHOD_MAP[method]


# =============================================================================
# MAKE ENV
# =============================================================================

def make_env(
    env_name:       str,
    method:         str,
    seed:           int,
    n_envs:         int,
    alpha:          float = 0.01,
    vecnorm_path:   Optional[str] = None,
    training:       bool = True,
) -> VecNormalize:
    """
    Create a normalized vectorized environment with optional PIRL wrapper.

    All environments are wrapped with VecNormalize regardless of domain.
    VecNormalize normalizes observations and rewards using running statistics.

    Parameters
    ----------
    env_name     : Gymnasium environment id (e.g. "CartPole-v1")
    method       : PIRL method — RL_PURE | PIRL_REWARD | PIRL_STATE | PIRL_POLICY
    seed         : Random seed for the environment
    n_envs       : Number of parallel environments
    alpha        : Physics prior weight alpha (from tuning, ignored for
                   RL_PURE and PIRL_POLICY)
    vecnorm_path : Path to saved VecNormalize stats (.pkl).
                   - None  → create fresh VecNormalize (tuning and training)
                   - str   → load saved stats (evaluation)
    training     : True  → VecNormalize updates running stats (tuning/training)
                   False → VecNormalize stats are frozen, norm_reward=False
                           (evaluation: model must see same scale as training)

    Returns
    -------
    VecNormalize wrapping a SubprocVecEnv (n_envs > 1) or DummyVecEnv (n_envs=1)

    Notes
    -----
    - PIRL_STATE wrapper is applied inside _init() so that VecNormalize
      sees the augmented observation space (phi normalized alongside state).
    - When loading saved stats for evaluation, pass training=False to prevent
      VecNormalize from updating statistics on evaluation data.
    """

    phi = get_phi(env_name) if method in _WRAPPED_METHODS else None

    def _init() -> gym.Env:
        env = gym.make(env_name)

        # Apply PIRL wrapper before VecNormalize
        # so that phi is normalized alongside the original state variables
        if method in _WRAPPED_METHODS:
            env = make_wrapper(env, method, phi, alpha)

        return env

    # ── Vectorize ─────────────────────────────────────────────────────────────
    vec_env_cls = DummyVecEnv #if n_envs == 1 else SubprocVecEnv

    venv = make_vec_env(
        _init,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
    )

    # ── VecNormalize ──────────────────────────────────────────────────────────
    if vecnorm_path is not None:
        # Evaluation: load saved running statistics and freeze them
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training    = False
        venv.norm_reward = False
    else:
        # Tuning / Training: fresh VecNormalize, statistics updated online
        venv = VecNormalize(venv, norm_obs=True, norm_reward=training)

    return venv