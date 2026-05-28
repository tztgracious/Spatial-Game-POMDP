"""
Self-play training infrastructure.

SelfPlayWrapper  — adapts the 2-agent ArenaEnv into a single-agent
                   Gymnasium interface so SB3's PPO can train agent 0
                   while agent 1's actions come from a fixed opponent policy.

ModelPool        — bounded checkpoint pool with probabilistic historical
                   sampling to prevent catastrophic forgetting (GUIDE §Phase 2).

run_self_play    — top-level training loop: alternate between learning and
                   swapping in a freshly sampled opponent each generation.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from agents.ppo_agent import make_ppo_agent
from env.arena import ArenaEnv, N_ACTIONS, OBS_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SelfPlayWrapper
# ---------------------------------------------------------------------------

class SelfPlayWrapper(gym.Wrapper):
    """
    Single-agent view of ArenaEnv for SB3 compatibility.

    The learner is always agent 0; agent 1 (the opponent) acts according to
    a fixed policy supplied via set_opponent().  Passing opponent=None falls
    back to uniform-random actions, which is the default for generation 0
    before any checkpoint exists.

    Observation space : Box(OBS_SIZE,) — agent-0's 7-element block only.
    Action space      : Discrete(N_ACTIONS) — single action for agent 0.
    """

    def __init__(self, env: ArenaEnv, opponent: Optional[PPO] = None) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=env.observation_space.low[:OBS_SIZE],
            high=env.observation_space.high[:OBS_SIZE],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._opponent: Optional[PPO] = opponent
        self._last_opp_obs: np.ndarray = np.zeros(OBS_SIZE, dtype=np.float32)

    # --- Gymnasium API ---

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        full_obs, info = self.env.reset(**kwargs)
        self._last_opp_obs = full_obs[OBS_SIZE:].copy()
        return full_obs[:OBS_SIZE].copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        opp_action = self._get_opponent_action()
        full_obs, reward, terminated, truncated, info = self.env.step(
            [int(action), opp_action]
        )
        self._last_opp_obs = full_obs[OBS_SIZE:].copy()
        return full_obs[:OBS_SIZE].copy(), reward, terminated, truncated, info

    # --- Public helpers ---

    def set_opponent(self, opponent: Optional[PPO]) -> None:
        """Swap the opponent policy (None → random actions)."""
        self._opponent = opponent

    # --- Internal ---

    def _get_opponent_action(self) -> int:
        if self._opponent is None:
            return int(self.env.action_space.sample()[1])
        action, _ = self._opponent.predict(self._last_opp_obs, deterministic=False)
        return int(action)


# ---------------------------------------------------------------------------
# ModelPool
# ---------------------------------------------------------------------------

class ModelPool:
    """
    Bounded pool of PPO checkpoint paths for self-play opponent sampling.

    Sampling policy (GUIDE §Phase 2):
      80% → latest checkpoint (exploit current best opponent)
      20% → uniformly random historical checkpoint (prevent forgetting)

    When the pool holds only one entry both cases return that entry.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_size: int = 10,
        historical_prob: float = 0.2,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.historical_prob = historical_prob
        self._entries: List[Tuple[Path, int]] = []  # (path_stem, global_step)

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, model: PPO, step: int) -> Path:
        """
        Save model to disk and register in the pool.
        Evicts (and deletes) the oldest entry when max_size is exceeded.
        """
        path = self.checkpoint_dir / f"gen_{step:08d}"
        model.save(str(path))
        self._entries.append((path, step))
        if len(self._entries) > self.max_size:
            old_path, _ = self._entries.pop(0)
            zip_path = old_path.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
        return path

    def sample_path(self) -> Optional[Path]:
        """Return a checkpoint path according to the sampling policy."""
        if not self._entries:
            return None
        if len(self._entries) == 1 or random.random() > self.historical_prob:
            return self._entries[-1][0]
        # Exclude latest so historical draw is meaningfully different
        return random.choice(self._entries[:-1])[0]

    def load_opponent(self) -> Optional[PPO]:
        """Sample a checkpoint and deserialise it as a PPO model for inference."""
        path = self.sample_path()
        if path is None:
            return None
        zip_path = path.with_suffix(".zip")
        load_path = str(zip_path if zip_path.exists() else path)
        return PPO.load(load_path, device="cpu")


# ---------------------------------------------------------------------------
# run_self_play
# ---------------------------------------------------------------------------

def run_self_play(
    total_timesteps: int = 500_000,
    generation_steps: int = 10_000,
    checkpoint_dir: str = "checkpoints",
    log_dir: Optional[str] = "runs/self_play",
    arena_kwargs: Optional[Dict[str, Any]] = None,
    ppo_kwargs: Optional[Dict[str, Any]] = None,
    pool_kwargs: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List] = None,
    seed: int = 0,
) -> PPO:
    """
    Self-play training loop.

    Each generation:
      1. Train the learner for `generation_steps` steps against the current opponent.
      2. Save a checkpoint to the model pool.
      3. Sample a new opponent (latest 80% / historical 20%).

    Generation 0 uses a random opponent (pool is empty).

    Parameters
    ----------
    total_timesteps : int
        Approximate total environment steps across all generations.
        Actual steps may exceed this slightly because SB3 always completes
        a full rollout buffer (n_steps) before updating.
    generation_steps : int
        Target steps per generation. Should be >= ppo n_steps to ensure
        at least one PPO update per generation.
    checkpoint_dir : str
        Directory for saved model checkpoints.
    log_dir : str or None
        TensorBoard log directory.  Pass None to disable logging.
    arena_kwargs : dict, optional
        Forwarded to ArenaEnv constructor.
    ppo_kwargs : dict, optional
        Override defaults in make_ppo_agent (e.g. n_steps=64 for tests).
    pool_kwargs : dict, optional
        Forwarded to ModelPool constructor (max_size, historical_prob).
    callbacks : list of BaseCallback, optional
        SB3 callbacks passed to every model.learn() call — use this to attach
        CurriculumCallback for Phase 3 reward shaping decay.
    seed : int
        RNG seed for the learner.

    Returns
    -------
    PPO
        The final trained model.
    """
    arena_kwargs = arena_kwargs or {}
    ppo_kwargs = ppo_kwargs or {}
    pool_kwargs = pool_kwargs or {}

    base_env = ArenaEnv(**arena_kwargs)
    env = SelfPlayWrapper(base_env, opponent=None)

    model = make_ppo_agent(env, log_dir=log_dir, seed=seed, **ppo_kwargs)
    pool = ModelPool(checkpoint_dir=checkpoint_dir, **pool_kwargs)

    n_generations = max(1, total_timesteps // generation_steps)
    logger.info(
        "Self-play start | generations=%d | steps_per_gen=%d | total≈%d",
        n_generations, generation_steps, total_timesteps,
    )

    for gen in range(n_generations):
        steps_so_far = gen * generation_steps
        logger.info(
            "Generation %d/%d | steps_so_far=%d | pool_size=%d",
            gen, n_generations, steps_so_far, pool.size,
        )

        model.learn(
            total_timesteps=generation_steps,
            reset_num_timesteps=(gen == 0),
            tb_log_name="self_play",
            progress_bar=False,
            callback=callbacks or [],
        )

        pool.add(model, steps_so_far + generation_steps)
        env.set_opponent(pool.load_opponent())

    logger.info("Self-play complete.")
    return model
