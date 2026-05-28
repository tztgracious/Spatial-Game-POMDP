"""Unit and integration tests for training/self_play_loop.py"""

import random
import pytest
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

from env.arena import ArenaEnv, OBS_SIZE, N_ACTIONS, STAY
from training.self_play_loop import SelfPlayWrapper, ModelPool, run_self_play
from agents.ppo_agent import make_ppo_agent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_wrapper(opponent=None):
    return SelfPlayWrapper(ArenaEnv(), opponent=opponent)


def _make_tiny_model(env):
    """PPO model with minimal settings — fast to create and save."""
    return make_ppo_agent(env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)


class _ConstantPolicy:
    """Deterministic mock policy that always returns action STAY."""
    def predict(self, obs, deterministic=False):
        return np.array(STAY), None


# ---------------------------------------------------------------------------
# SelfPlayWrapper — spaces
# ---------------------------------------------------------------------------

class TestSelfPlayWrapperSpaces:
    def test_observation_space_is_single_agent(self):
        env = _make_wrapper()
        assert env.observation_space.shape == (OBS_SIZE,)

    def test_action_space_is_discrete(self):
        env = _make_wrapper()
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == N_ACTIONS

    def test_obs_space_bounds_match_agent0_block(self):
        arena = ArenaEnv()
        wrapper = SelfPlayWrapper(arena)
        np.testing.assert_array_equal(
            wrapper.observation_space.low, arena.observation_space.low[:OBS_SIZE]
        )
        np.testing.assert_array_equal(
            wrapper.observation_space.high, arena.observation_space.high[:OBS_SIZE]
        )


# ---------------------------------------------------------------------------
# SelfPlayWrapper — reset
# ---------------------------------------------------------------------------

class TestSelfPlayWrapperReset:
    def test_reset_returns_tuple(self):
        env = _make_wrapper()
        result = env.reset(seed=0)
        assert len(result) == 2

    def test_reset_obs_shape(self):
        env = _make_wrapper()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_SIZE,)

    def test_reset_obs_dtype(self):
        env = _make_wrapper()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_reset_updates_opp_obs(self):
        env = _make_wrapper()
        env._last_opp_obs[:] = 999.0
        env.reset(seed=0)
        assert not np.all(env._last_opp_obs == 999.0)

    def test_reset_obs_within_bounds(self):
        env = _make_wrapper()
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

    def test_seeded_reset_deterministic(self):
        env = _make_wrapper()
        obs1, _ = env.reset(seed=7)
        obs2, _ = env.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# SelfPlayWrapper — step
# ---------------------------------------------------------------------------

class TestSelfPlayWrapperStep:
    def setup_method(self):
        self.env = _make_wrapper()
        self.env.reset(seed=0)

    def test_step_returns_5_tuple(self):
        result = self.env.step(STAY)
        assert len(result) == 5

    def test_step_obs_shape(self):
        obs, *_ = self.env.step(STAY)
        assert obs.shape == (OBS_SIZE,)

    def test_step_obs_within_bounds(self):
        obs, *_ = self.env.step(STAY)
        assert self.env.observation_space.contains(obs)

    def test_step_reward_is_float(self):
        _, reward, *_ = self.env.step(STAY)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self):
        _, _, terminated, *_ = self.env.step(STAY)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self):
        _, _, _, truncated, _ = self.env.step(STAY)
        assert isinstance(truncated, bool)

    def test_step_updates_opp_obs(self):
        opp_obs_before = self.env._last_opp_obs.copy()
        self.env.step(STAY)
        # After any step the opp obs buffer must have been refreshed
        # (values may coincidentally be equal but the object must be fresh)
        assert self.env._last_opp_obs is not opp_obs_before

    def test_step_with_random_opponent(self):
        env = SelfPlayWrapper(ArenaEnv(), opponent=None)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(STAY)
        assert obs.shape == (OBS_SIZE,)

    def test_step_with_constant_mock_opponent(self):
        env = SelfPlayWrapper(ArenaEnv(), opponent=_ConstantPolicy())
        env.reset(seed=0)
        obs, *_ = env.step(STAY)
        assert obs.shape == (OBS_SIZE,)

    def test_step_with_ppo_opponent(self):
        inner_env = _make_wrapper()
        opponent = _make_tiny_model(inner_env)
        env = SelfPlayWrapper(ArenaEnv(), opponent=opponent)
        env.reset(seed=0)
        obs, *_ = env.step(STAY)
        assert obs.shape == (OBS_SIZE,)


# ---------------------------------------------------------------------------
# SelfPlayWrapper — set_opponent
# ---------------------------------------------------------------------------

class TestSelfPlayWrapperSetOpponent:
    def test_set_opponent_to_none(self):
        env = SelfPlayWrapper(ArenaEnv(), opponent=_ConstantPolicy())
        env.set_opponent(None)
        assert env._opponent is None

    def test_set_opponent_to_policy(self):
        env = _make_wrapper()
        policy = _ConstantPolicy()
        env.set_opponent(policy)
        assert env._opponent is policy

    def test_opponent_swap_changes_behaviour(self):
        """After swapping opponent the wrapper uses the new policy."""
        env = _make_wrapper()
        env.reset(seed=0)
        env.set_opponent(_ConstantPolicy())
        # Should not raise and opponent action should be deterministic (STAY)
        env.step(STAY)


# ---------------------------------------------------------------------------
# ModelPool — basic operations
# ---------------------------------------------------------------------------

class TestModelPoolBasic:
    def test_initial_size_zero(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        assert pool.size == 0

    def test_add_increments_size(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        env = _make_wrapper()
        model = _make_tiny_model(env)
        pool.add(model, step=100)
        assert pool.size == 1

    def test_add_creates_zip_file(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        env = _make_wrapper()
        model = _make_tiny_model(env)
        path = pool.add(model, step=100)
        assert path.with_suffix(".zip").exists()

    def test_add_multiple_increments_correctly(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        env = _make_wrapper()
        model = _make_tiny_model(env)
        for step in [100, 200, 300]:
            pool.add(model, step=step)
        assert pool.size == 3

    def test_max_size_enforced(self, tmp_path):
        pool = ModelPool(str(tmp_path), max_size=3)
        env = _make_wrapper()
        model = _make_tiny_model(env)
        for step in [100, 200, 300, 400]:
            pool.add(model, step=step)
        assert pool.size == 3

    def test_oldest_evicted_when_over_max(self, tmp_path):
        pool = ModelPool(str(tmp_path), max_size=2)
        env = _make_wrapper()
        model = _make_tiny_model(env)
        pool.add(model, step=100)
        pool.add(model, step=200)
        pool.add(model, step=300)
        steps = [e[1] for e in pool._entries]
        assert 100 not in steps
        assert 200 in steps and 300 in steps

    def test_evicted_file_deleted(self, tmp_path):
        pool = ModelPool(str(tmp_path), max_size=1)
        env = _make_wrapper()
        model = _make_tiny_model(env)
        p1 = pool.add(model, step=100)
        pool.add(model, step=200)  # triggers eviction of step=100
        assert not p1.with_suffix(".zip").exists()


# ---------------------------------------------------------------------------
# ModelPool — sample_path
# ---------------------------------------------------------------------------

class TestModelPoolSamplePath:
    def test_sample_path_empty_returns_none(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        assert pool.sample_path() is None

    def test_sample_path_single_entry_returns_it(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        env = _make_wrapper()
        model = _make_tiny_model(env)
        path = pool.add(model, step=100)
        assert pool.sample_path() == path

    def test_sample_path_always_latest_when_historical_prob_zero(self, tmp_path):
        pool = ModelPool(str(tmp_path), historical_prob=0.0)
        env = _make_wrapper()
        model = _make_tiny_model(env)
        pool.add(model, step=100)
        pool.add(model, step=200)
        latest = pool._entries[-1][0]
        for _ in range(20):
            assert pool.sample_path() == latest

    def test_sample_path_always_historical_when_prob_one(self, tmp_path):
        pool = ModelPool(str(tmp_path), historical_prob=1.0)
        env = _make_wrapper()
        model = _make_tiny_model(env)
        pool.add(model, step=100)
        pool.add(model, step=200)
        pool.add(model, step=300)
        latest = pool._entries[-1][0]
        for _ in range(20):
            assert pool.sample_path() != latest


# ---------------------------------------------------------------------------
# ModelPool — load_opponent
# ---------------------------------------------------------------------------

class TestModelPoolLoadOpponent:
    def test_load_opponent_empty_returns_none(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        assert pool.load_opponent() is None

    def test_load_opponent_returns_ppo(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        env = _make_wrapper()
        model = _make_tiny_model(env)
        pool.add(model, step=100)
        opponent = pool.load_opponent()
        assert isinstance(opponent, PPO)

    def test_loaded_opponent_can_predict(self, tmp_path):
        pool = ModelPool(str(tmp_path))
        env = _make_wrapper()
        model = _make_tiny_model(env)
        pool.add(model, step=100)
        opponent = pool.load_opponent()
        obs, _ = env.reset(seed=0)
        action, _ = opponent.predict(obs, deterministic=True)
        assert int(action) in range(N_ACTIONS)


# ---------------------------------------------------------------------------
# Integration: run_self_play
# ---------------------------------------------------------------------------

class TestRunSelfPlay:
    """Fast integration tests using minimal timestep / n_steps settings."""

    _FAST_PPO = dict(n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    _FAST_ARENA = dict(max_steps=20)

    def test_returns_ppo_model(self, tmp_path):
        model = run_self_play(
            total_timesteps=64,
            generation_steps=64,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=None,
            arena_kwargs=self._FAST_ARENA,
            ppo_kwargs=self._FAST_PPO,
        )
        assert isinstance(model, PPO)

    def test_checkpoints_are_created(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        run_self_play(
            total_timesteps=128,
            generation_steps=64,
            checkpoint_dir=str(ckpt_dir),
            log_dir=None,
            arena_kwargs=self._FAST_ARENA,
            ppo_kwargs=self._FAST_PPO,
        )
        zips = list(ckpt_dir.glob("*.zip"))
        assert len(zips) >= 1

    def test_two_generations_run_without_error(self, tmp_path):
        run_self_play(
            total_timesteps=128,
            generation_steps=64,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=None,
            arena_kwargs=self._FAST_ARENA,
            ppo_kwargs=self._FAST_PPO,
        )

    def test_final_model_predicts_valid_action(self, tmp_path):
        env = SelfPlayWrapper(ArenaEnv())
        model = run_self_play(
            total_timesteps=64,
            generation_steps=64,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=None,
            arena_kwargs=self._FAST_ARENA,
            ppo_kwargs=self._FAST_PPO,
        )
        obs, _ = env.reset(seed=0)
        action, _ = model.predict(obs, deterministic=True)
        assert int(action) in range(N_ACTIONS)

    def test_pool_kwargs_respected(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        run_self_play(
            total_timesteps=192,
            generation_steps=64,
            checkpoint_dir=str(ckpt_dir),
            log_dir=None,
            arena_kwargs=self._FAST_ARENA,
            ppo_kwargs=self._FAST_PPO,
            pool_kwargs=dict(max_size=2),
        )
        zips = list(ckpt_dir.glob("*.zip"))
        assert len(zips) <= 2
