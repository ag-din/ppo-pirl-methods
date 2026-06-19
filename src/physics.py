"""
Physics priors (φ) for supported environments.

"""

import numpy as np
from typing import Optional


def phi_mountaincar(obs: np.ndarray, action: Optional[np.ndarray] = None) -> float:
    """
    Penalty for MountainCarContinuous-v0.

    Uses the horizontal position and returns squared distance
    to the goal location. Larger values indicate the car is farther
    from the goal (undesirable).

    """

    car_pos = obs[0]
    goal_pos = 0.45
    return float((car_pos - goal_pos) ** 2)


def phi_reacher(obs: np.ndarray, action: Optional[np.ndarray] = None) -> float:
    """
    Penalty for Reacher-v5.

    This simple prior penalizes high joint angular velocities to favor
    smoother movements.
    """

    ang_vel_1 = obs[6]
    ang_vel_2 = obs[7]

    return float(ang_vel_1 ** 2 + ang_vel_2 ** 2)


def phi_acrobot(obs: np.ndarray, action: Optional[np.ndarray] = None) -> float:
    """
    Penalty for Acrobot-v1.

    Observations provide (cos(theta1), sin(theta1), cos(theta2),
    sin(theta2), ...). We recover the two joint angles with
    arctan2 and compute a normalized distance-like term `d` that
    measures how far the end-effector is from a reference. The returned
    squared ratio is bounded and normalized by `d_max`.
    """

    theta1 = np.arctan2(obs[1], obs[0])
    theta2 = np.arctan2(obs[3], obs[2])
    d_max = 2.0

    d = np.abs(-np.cos(theta1) - np.cos(theta1 + theta2))

    return float((d / d_max) ** 2)


def phi_cartpole(obs, action=None):
    """
    Penalty for CartPole-v1.

    This prior penalizes large pole angles and angular velocities, which
    correspond to physically unstable states. The returned value is the
    sum of the squared pole angle and angular velocity, encouraging the
    agent to keep the pole upright and stable.
    """

    pole_angle = obs[2]
    pole_ang_vel = obs[3]

    return float(pole_ang_vel**2 + pole_angle**2)


# Registry — env_name → φ function
PHI_REGISTRY = {
    "MountainCarContinuous-v0":   phi_mountaincar,
    "Reacher-v5":                 phi_reacher,
    "Acrobot-v1":                 phi_acrobot,    
    "CartPole-v1":                phi_cartpole,
}

def get_phi(env_name: str):
    """
    Retrieve the φ function for a given environment name.

    Parameters
    ----------
    env_name : str
        The name of the environment (e.g., "MountainCarContinuous-v0").
    
    Returns
    -------
    phi_fn : callable(obs, action) → float
        The physics prior function corresponding to the environment.
    """
    if env_name not in PHI_REGISTRY:
        raise ValueError(f"No φ defined for env: {env_name}")
    return PHI_REGISTRY[env_name]