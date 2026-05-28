"""Unit tests for agents/ppo_agent.py"""

import pytest
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from env.arena import ArenaEnv
from training.self_play_loop import SelfPlayWrapper
from agents.ppo_agent import make_ppo_agent, _DEFAULT_PPO_KWARGS


@pytest.fixture
def single_agent_env():
    return SelfPlayWrapper(ArenaEnv())


class TestMakePPOAgent:
    def test_returns_ppo_instance(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, verbose=0)
        assert isinstance(model, PPO)

    def test_default_net_arch(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, verbose=0)
        net_arch = model.policy.net_arch
        assert net_arch == [128, 128]

    def test_override_net_arch(self, single_agent_env):
        model = make_ppo_agent(
            single_agent_env,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0,
        )
        assert model.policy.net_arch == [64, 64]

    def test_override_learning_rate(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, learning_rate=1e-3, verbose=0)
        assert model.learning_rate == 1e-3

    def test_seed_is_set(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, seed=42, verbose=0)
        assert model.seed == 42

    def test_tensorboard_log_none_by_default(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, verbose=0)
        assert model.tensorboard_log is None

    def test_tensorboard_log_set(self, single_agent_env, tmp_path):
        model = make_ppo_agent(single_agent_env, log_dir=str(tmp_path), verbose=0)
        assert model.tensorboard_log == str(tmp_path)

    def test_policy_observation_space_matches_env(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, verbose=0)
        assert model.observation_space == single_agent_env.observation_space

    def test_policy_action_space_matches_env(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, verbose=0)
        assert model.action_space == single_agent_env.action_space

    def test_predict_returns_valid_action(self, single_agent_env):
        model = make_ppo_agent(single_agent_env, verbose=0)
        obs, _ = single_agent_env.reset(seed=0)
        action, state = model.predict(obs, deterministic=True)
        assert int(action) in range(6)
        assert state is None
