# =============================================================================
# callbacks.py — SB3 callbacks for PIRL integration methods
# =============================================================================
#
# PIRL_POLICY modifies the PPO training loss by adding a physics penalty:
#
#   L = L_PPO + α · E[φ(s, a)]
#
# This cannot be implemented as a Gymnasium wrapper because it operates
# on the training loss rather than on the environment interface.
# It is implemented as a Stable-Baselines3 callback that hooks into
# the rollout buffer after each data collection phase.
#
# How it works:
#   After each rollout, PhysicsPolicyCallback computes φ(s, a) for every
#   (observation, action) pair in the rollout buffer and subtracts
#   α · φ from the returns and advantages. This is equivalent to adding
#   α · E[φ(s, a)] to the PPO loss because PPO maximizes the expected
#   advantage — penalizing the returns directly penalizes the loss.
#
# =============================================================================

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Callable

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# TYPE ALIAS
# =============================================================================

PhiFn = Callable[[np.ndarray, np.ndarray], float]
# φ(obs, action) → scalar physics prior value


# =============================================================================
# PHYSICS POLICY CALLBACK
# =============================================================================

# =============================================================================
# callbacks.py — SB3 callbacks for PIRL integration methods
# =============================================================================

from typing import Callable
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

PhiFn = Callable[[np.ndarray, np.ndarray], float]


class PhysicsPolicyCallback(BaseCallback):
    """
    PIRL_POLICY: augments PPO training loss with a physics penalty.

        L = L_PPO + α · E[φ(s, a)]

    φ is normalized to the same scale as the raw advantages before
    the PPO internal normalization, so α has a consistent interpretation
    across environments: α = 0.1 means the prior contributes 10% of
    the std of the advantages.

    Parameters
    ----------
    phi_fn : callable(obs, action) → float
    alpha  : float
    verbose : int
    """

    def __init__(self, phi_fn: PhiFn, alpha: float, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.phi_fn = phi_fn
        self.alpha  = alpha

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buffer = self.model.rollout_buffer

        observations = buffer.observations  # (n_steps, n_envs, obs_dim)
        actions      = buffer.actions       # (n_steps, n_envs, act_dim)
        n_steps, n_envs = actions.shape[:2]

        # ── Compute φ(s, a) for every transition ──────────────────────────
        phi_penalties = np.zeros((n_steps, n_envs), dtype=np.float32)
        for t in range(n_steps):
            for e in range(n_envs):
                phi_penalties[t, e] = float(self.phi_fn(observations[t, e], actions[t, e]))

        # ── Normalize φ to the scale of raw advantages ────────────────────
        # SB3 normalizes advantages AFTER this callback, so we anchor φ
        # to the current std of advantages so that alpha stays interpretable.
        #phi_norm = (phi_penalties - phi_penalties.mean()) / (phi_penalties.std() + 1e-8)
        #penalty  = self.alpha * phi_norm * buffer.advantages.std()  # (n_steps, n_envs)

        # no norm
        penalty = self.alpha * phi_penalties


        # ── Subtract penalty from returns and advantages ───────────────────
        buffer.returns    -= penalty
        buffer.advantages -= penalty

        if self.verbose >= 1:
            print(
                f"  [PhysicsPolicyCallback] "
                f"mean φ = {phi_penalties.mean():.4f}  |  "
                f"std φ  = {phi_penalties.std():.4f}  |  "
                f"mean penalty = {penalty.mean():.4f}  |  "
                f"α = {self.alpha}"
            )