"""Integration tests: multi-module end-to-end flows."""

import pytest
import numpy as np
from env.arena import ArenaEnv, FIRE, STAY
from core.geometry import has_line_of_sight


class TestFullEpisode:
    def test_random_episode_completes(self):
        env = ArenaEnv(max_steps=200)
        obs, _ = env.reset(seed=7)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert obs.shape == env.observation_space.shape
            assert env.observation_space.contains(obs), "Obs outside declared bounds"
        assert steps > 0

    def test_episode_always_ends(self):
        """No episode should run forever regardless of actions."""
        env = ArenaEnv(max_steps=100)
        for seed in range(10):
            env.reset(seed=seed)
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert terminated or truncated

    def test_multiple_resets_clean_state(self):
        env = ArenaEnv(walls=[], fire_damage=25.0, max_hp=100.0)
        for seed in range(5):
            env.reset(seed=seed)
            assert env._step_count == 0
            np.testing.assert_array_equal(env._hp, [env.max_hp, env.max_hp])

    def test_rewards_sum_to_zero_on_kill(self):
        """Kill/death rewards are symmetric; net transfer should be zero."""
        env = ArenaEnv(walls=[], fire_damage=50.0, max_hp=100.0, kill_reward=10.0)
        env.reset(seed=0)
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        _, _, _, _, info0 = env.step([FIRE, STAY])
        _, _, _, _, info1 = env.step([FIRE, STAY])
        # rewards_both sums to: kill_reward - kill_reward + 2*(-time_penalty)
        r_both = info1["rewards_both"]
        assert r_both[0] + r_both[1] == pytest.approx(-2 * env.time_penalty, abs=1e-5)

    def test_los_consistent_with_geometry(self):
        """Info LoS flag must match a direct geometry query."""
        walls = [((5.0, 2.0), (5.0, 8.0))]
        env = ArenaEnv(walls=walls, max_steps=50)
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            direct_los = has_line_of_sight(
                tuple(env._positions[0]), tuple(env._positions[1]), walls
            )
            assert info["los"] == direct_los
            if terminated or truncated:
                env.reset(seed=42)

    def test_obs_matches_env_state(self):
        """Observation vector values must correspond to env._positions and _hp."""
        env = ArenaEnv(walls=[], max_steps=20)
        env.reset(seed=3)
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            # Agent-0: obs[0]=x, obs[1]=y, obs[2]=hp_norm
            assert obs[0] == pytest.approx(env._positions[0][0])
            assert obs[1] == pytest.approx(env._positions[0][1])
            assert obs[2] == pytest.approx(env._hp[0] / env.max_hp)
            if terminated or truncated:
                env.reset()
