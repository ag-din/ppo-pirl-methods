# =============================================================================
# wrappers.py — PIRL integration method wrappers for Gymnasium environments
# =============================================================================
#
# Implements four PIRL integration strategies as Gymnasium wrappers:
#
#   PIRL_REWARD  → RewardShapingWrapper     r' = r - α · φ(s, a)
#   PIRL_STATE   → StateAugmentationWrapper  s' = [s, φ(s, a)]
#   PIRL_POLICY  → no wrapper (handled via SB3 callback in train.py)
#   RL_PURE      → no wrapper (identity)
#
# Usage:
#   from src.wrappers import make_wrapper
#   from src.physics import get_phi
#
#   phi_fn = get_phi(env_name)
#   env    = make_wrapper(env, method, phi_fn, alpha)
#
# =============================================================================

# ── Libraires ──────────────────────────────────────────────────────────
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable


# =============================================================================
# TYPE ALIAS
# =============================================================================

PhiFn = Callable[[np.ndarray, np.ndarray], float]
# φ(obs, action) → scalar physics prior value


# =============================================================================
# REWARD SHAPING WRAPPER
# =============================================================================

class RewardShapingWrapper(gym.RewardWrapper):
    """
    PIRL_REWARD: modifies the reward signal at each step.

        r'(s, a) = r(s, a) - α · φ(s, a)

    The physics prior φ penalizes physically undesirable states or actions.
    A higher φ value means a larger penalty, guiding the agent away from
    physically problematic behaviors.

    The observation and action spaces are unchanged.

    Parameters
    ----------
    env   : Gymnasium environment
    phi   : callable(obs, action) → float
    alpha : float, weight of the physics penalty (tuned per environment)
    """

    def __init__(self, env: gym.Env, phi: PhiFn, alpha: float) -> None:
        super().__init__(env)
        self.phi   = phi
        self.alpha = alpha
        self._last_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute the physics prior on the transitioned state and action.
        # This value penalizes bad physics behavior and is subtracted from
        # the environment reward to shape agent learning.
        phi_value = float(self.phi(obs, np.asarray(action)))
        modified_reward = reward - self.alpha * phi_value

        # Preserve diagnostics so downstream analysis can inspect the
        # original reward and the prior contribution separately.
        info["phi"] = phi_value
        info["reward_original"] = reward
        self._last_obs = obs
        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info


# =============================================================================
# STATE AUGMENTATION WRAPPER
# =============================================================================

class StateAugmentationWrapper(gym.ObservationWrapper):
    """
    PIRL_STATE: augments the observation with the physics prior value.

        s'(s, a) = [s, φ(s, a)]

    The observation space is extended by one dimension to include φ.
    This wrapper must be applied BEFORE VecNormalize so that φ is
    normalized alongside the original state variables.

    Note: because φ depends on the action taken at the previous step,
    φ is computed from the transition (s_prev → a → s_next) and
    appended to s_next. At reset, φ is initialized to 0.

    Parameters
    ----------
    env : Gymnasium environment
    phi : callable(obs, action) → float
    """

    def __init__(self, env: gym.Env, phi: PhiFn) -> None:
        super().__init__(env)
        self.phi        = phi
        self._last_action = None
        self._phi_value   = 0.0

        low  = np.append(env.observation_space.low,  -np.inf).astype(np.float32)
        high = np.append(env.observation_space.high,  np.inf).astype(np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Append current φ value to the observation."""
        return np.append(obs, self._phi_value)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute φ based on the observed next state and the taken action.
        # The augmented observation includes the prior from the current transition.
        self._phi_value = float(self.phi(obs, np.asarray(action)))
        self._last_action = action
        info["phi"] = self._phi_value
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize φ to zero at the start of an episode because no action
        # has been taken yet to define a previous transition.
        self._phi_value = 0.0
        self._last_action = None
        return self.observation(obs), info


# =============================================================================
# FACTORY
# =============================================================================

def make_wrapper(
    env:    gym.Env,
    method: str,
    phi:    PhiFn,
    alpha:  float = 0.01,
) -> gym.Env:
    """
    Factory: returns the appropriate PIRL wrapper for the given method.

    Parameters
    ----------
    env    : Gymnasium environment (already instantiated)
    method : str — one of RL_PURE, PIRL_REWARD, PIRL_STATE, PIRL_POLICY
    phi    : callable(obs, action) → float — physics prior φ(s, a)
    alpha  : float — weight of the physics prior

    Returns
    -------
    Wrapped (or unwrapped) Gymnasium environment.

    Notes
    -----
    - PIRL_POLICY does not use a wrapper — physics integration is
      handled via a SB3 callback in train.py.
    - For PIRL_STATE, apply this wrapper BEFORE VecNormalize.
    """

    if method == "RL_PURE":
        return env

    elif method == "PIRL_REWARD":
        return RewardShapingWrapper(env, phi, alpha)

    elif method == "PIRL_STATE":
        return StateAugmentationWrapper(env, phi)

    elif method == "PIRL_POLICY":
        # No wrapper — policy augmentation is handled via PhysicsPolicyCallback
        # in train.py. The environment is returned unchanged.
        return env

    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Expected one of: RL_PURE, PIRL_REWARD, PIRL_STATE, PIRL_POLICY."
        )
