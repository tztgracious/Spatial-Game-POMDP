"""
PPO agent factory for the arena self-play setup.

Wraps stable-baselines3 PPO with sensible defaults for a 7-dim observation
space and a 6-action discrete policy (MLP 128×128).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import PPO

# Defaults match the GUIDE spec: MLP [128, 128] Actor-Critic
_DEFAULT_PPO_KWARGS: Dict[str, Any] = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[128, 128]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)


def make_ppo_agent(
    env: gym.Env,
    log_dir: Optional[str] = None,
    seed: int = 0,
    **overrides: Any,
) -> PPO:
    """
    Create a PPO agent configured for the self-play arena.

    Any keyword argument overrides the corresponding default, allowing
    callers to shrink n_steps / batch_size for fast unit-test runs.
    """
    kwargs = {**_DEFAULT_PPO_KWARGS, **overrides}
    kwargs["env"] = env
    kwargs["seed"] = seed
    kwargs["tensorboard_log"] = log_dir
    return PPO(**kwargs)
