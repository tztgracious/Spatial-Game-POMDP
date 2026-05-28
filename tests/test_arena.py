"""Unit tests for env/arena.py"""

import pytest
import numpy as np
from env.arena import ArenaEnv, STAY, UP, DOWN, LEFT, RIGHT, FIRE, OBS_SIZE, N_AGENTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_env():
    env = ArenaEnv()
    env.reset(seed=0)
    return env


@pytest.fixture
def open_env():
    """Arena with no walls — agents always have LoS."""
    env = ArenaEnv(walls=[])
    env.reset(seed=0)
    return env


@pytest.fixture
def full_wall_env():
    """Full vertical wall at x=5 — agents on opposite sides never have LoS."""
    env = ArenaEnv(walls=[((5.0, 0.0), (5.0, 10.0))])
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestArenaInit:
    def test_default_wall_created(self):
        env = ArenaEnv()
        assert len(env.walls) == 1

    def test_custom_walls_preserved(self):
        w = [((1.0, 1.0), (9.0, 1.0))]
        env = ArenaEnv(walls=w)
        assert env.walls == w

    def test_observation_space_shape(self):
        env = ArenaEnv()
        assert env.observation_space.shape == (N_AGENTS * OBS_SIZE,)

    def test_action_space_sizes(self):
        env = ArenaEnv()
        assert list(env.action_space.nvec) == [6, 6]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestArenaReset:
    def test_returns_obs_and_dict(self):
        env = ArenaEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_obs_shape_after_reset(self):
        env = ArenaEnv()
        obs, _ = env.reset()
        assert obs.shape == (N_AGENTS * OBS_SIZE,)

    def test_hp_full_after_reset(self):
        env = ArenaEnv(max_hp=100.0)
        env.reset()
        np.testing.assert_array_equal(env._hp, [100.0, 100.0])

    def test_step_counter_zero_after_reset(self):
        env = ArenaEnv()
        env.reset()
        assert env._step_count == 0

    def test_positions_within_bounds_after_reset(self):
        env = ArenaEnv()
        env.reset()
        for i in range(N_AGENTS):
            assert 0 <= env._positions[i][0] <= env.width
            assert 0 <= env._positions[i][1] <= env.height

    def test_reset_restores_hp_after_damage(self, open_env):
        open_env._positions[0] = [2.0, 5.0]
        open_env._positions[1] = [8.0, 5.0]
        open_env.step([FIRE, STAY])
        open_env.reset()
        np.testing.assert_array_equal(open_env._hp, [open_env.max_hp, open_env.max_hp])

    def test_seeded_reset_is_deterministic(self):
        env = ArenaEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

class TestArenaMovement:
    def test_stay_does_not_move(self, default_env):
        pos_before = default_env._positions.copy()
        default_env.step([STAY, STAY])
        np.testing.assert_array_equal(default_env._positions, pos_before)

    def test_move_up_increases_y(self, default_env):
        default_env._positions[0] = [5.0, 5.0]
        default_env.step([UP, STAY])
        assert default_env._positions[0][1] == pytest.approx(5.5)

    def test_move_down_decreases_y(self, default_env):
        default_env._positions[0] = [5.0, 5.0]
        default_env.step([DOWN, STAY])
        assert default_env._positions[0][1] == pytest.approx(4.5)

    def test_move_left_decreases_x(self, default_env):
        default_env._positions[0] = [5.0, 5.0]
        default_env.step([LEFT, STAY])
        assert default_env._positions[0][0] == pytest.approx(4.5)

    def test_move_right_increases_x(self, default_env):
        default_env._positions[0] = [5.0, 5.0]
        default_env.step([RIGHT, STAY])
        assert default_env._positions[0][0] == pytest.approx(5.5)

    def test_clamp_at_top_boundary(self, default_env):
        default_env._positions[0] = [5.0, default_env.height]
        default_env.step([UP, STAY])
        assert default_env._positions[0][1] <= default_env.height

    def test_clamp_at_bottom_boundary(self, default_env):
        default_env._positions[0] = [5.0, 0.0]
        default_env.step([DOWN, STAY])
        assert default_env._positions[0][1] >= 0.0

    def test_clamp_at_left_boundary(self, default_env):
        default_env._positions[0] = [0.0, 5.0]
        default_env.step([LEFT, STAY])
        assert default_env._positions[0][0] >= 0.0

    def test_clamp_at_right_boundary(self, default_env):
        default_env._positions[0] = [default_env.width, 5.0]
        default_env.step([RIGHT, STAY])
        assert default_env._positions[0][0] <= default_env.width

    def test_both_agents_move_independently(self, default_env):
        default_env._positions[0] = [3.0, 3.0]
        default_env._positions[1] = [7.0, 7.0]
        default_env.step([RIGHT, LEFT])
        assert default_env._positions[0][0] == pytest.approx(3.5)
        assert default_env._positions[1][0] == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------

class TestArenaCombat:
    def test_fire_with_los_deals_damage(self, open_env):
        open_env._positions[0] = [2.0, 5.0]
        open_env._positions[1] = [8.0, 5.0]
        hp_before = open_env._hp[1]
        open_env.step([FIRE, STAY])
        assert open_env._hp[1] < hp_before

    def test_fire_damage_amount(self):
        env = ArenaEnv(walls=[], fire_damage=25.0, max_hp=100.0)
        env.reset()
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        env.step([FIRE, STAY])
        assert env._hp[1] == pytest.approx(75.0)

    def test_fire_without_los_no_damage(self, full_wall_env):
        full_wall_env._positions[0] = [2.0, 5.0]
        full_wall_env._positions[1] = [8.0, 5.0]
        hp_before = full_wall_env._hp[1]
        full_wall_env.step([FIRE, STAY])
        assert full_wall_env._hp[1] == hp_before

    def test_fire_without_los_incurs_penalty(self, full_wall_env):
        full_wall_env._positions[0] = [2.0, 5.0]
        full_wall_env._positions[1] = [8.0, 5.0]
        _, reward, _, _, _ = full_wall_env.step([FIRE, STAY])
        # reward = -blind_fire_penalty - time_penalty
        assert reward < -full_wall_env.time_penalty

    def test_simultaneous_fire_both_take_damage(self, open_env):
        open_env._positions[0] = [2.0, 5.0]
        open_env._positions[1] = [8.0, 5.0]
        hp0_before, hp1_before = open_env._hp[0], open_env._hp[1]
        open_env.step([FIRE, FIRE])
        assert open_env._hp[0] < hp0_before
        assert open_env._hp[1] < hp1_before

    def test_hp_does_not_go_below_zero(self, open_env):
        open_env._positions[0] = [2.0, 5.0]
        open_env._positions[1] = [8.0, 5.0]
        open_env._hp[1] = 1.0  # nearly dead
        open_env.step([FIRE, STAY])
        assert open_env._hp[1] >= 0.0


# ---------------------------------------------------------------------------
# Termination and truncation
# ---------------------------------------------------------------------------

class TestArenaTermination:
    def test_kill_sets_terminated(self):
        env = ArenaEnv(walls=[], fire_damage=50.0, max_hp=100.0)
        env.reset()
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        env.step([FIRE, STAY])
        _, _, terminated, _, _ = env.step([FIRE, STAY])
        assert terminated

    def test_kill_gives_positive_reward_to_winner(self):
        env = ArenaEnv(walls=[], fire_damage=50.0, max_hp=100.0, kill_reward=10.0)
        env.reset()
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        env.step([FIRE, STAY])
        _, reward, terminated, _, info = env.step([FIRE, STAY])
        assert terminated
        assert reward > 0  # agent 0 is the winner

    def test_kill_gives_negative_reward_to_loser(self):
        env = ArenaEnv(walls=[], fire_damage=50.0, max_hp=100.0, kill_reward=10.0)
        env.reset()
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        env.step([STAY, FIRE])  # agent 1 fires
        _, reward, terminated, _, info = env.step([STAY, FIRE])
        assert terminated
        assert reward < 0  # agent 0 is the loser

    def test_max_steps_truncates(self):
        env = ArenaEnv(max_steps=5)
        env.reset()
        for _ in range(4):
            _, _, terminated, truncated, _ = env.step([STAY, STAY])
            assert not truncated
        _, _, _, truncated, _ = env.step([STAY, STAY])
        assert truncated

    def test_step_counter_increments(self, default_env):
        default_env.step([STAY, STAY])
        default_env.step([STAY, STAY])
        assert default_env._step_count == 2


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class TestArenaObservations:
    def test_obs_shape(self, default_env):
        obs, _, _, _, _ = default_env.step([STAY, STAY])
        assert obs.shape == (N_AGENTS * OBS_SIZE,)

    def test_obs_dtype_float32(self, default_env):
        obs, _, _, _, _ = default_env.step([STAY, STAY])
        assert obs.dtype == np.float32

    def test_obs_hp_normalised(self, open_env):
        obs, _ = open_env.reset()
        # own_hp is index 2 (block 0) and index 9 (block 1)
        assert obs[2] == pytest.approx(1.0)
        assert obs[9] == pytest.approx(1.0)

    def test_opp_hidden_when_no_los(self, full_wall_env):
        full_wall_env._positions[0] = [2.0, 5.0]
        full_wall_env._positions[1] = [8.0, 5.0]
        obs, _, _, _, _ = full_wall_env.step([STAY, STAY])
        # Agent-0 block: [own_x, own_y, own_hp, los, opp_x, opp_y, opp_hp]
        assert obs[3] == pytest.approx(0.0)  # los = False
        assert obs[4] == pytest.approx(0.0)  # opp_x hidden
        assert obs[5] == pytest.approx(0.0)  # opp_y hidden
        assert obs[6] == pytest.approx(0.0)  # opp_hp hidden

    def test_opp_visible_when_los(self, open_env):
        open_env._positions[0] = [2.0, 5.0]
        open_env._positions[1] = [8.0, 5.0]
        obs, _, _, _, _ = open_env.step([STAY, STAY])
        assert obs[3] == pytest.approx(1.0)   # los = True
        assert obs[4] == pytest.approx(8.0)   # opp_x
        assert obs[5] == pytest.approx(5.0)   # opp_y

    def test_info_contains_los(self, default_env):
        _, _, _, _, info = default_env.step([STAY, STAY])
        assert "los" in info
        assert isinstance(info["los"], bool)

    def test_info_contains_hp(self, default_env):
        _, _, _, _, info = default_env.step([STAY, STAY])
        assert "hp" in info
        assert info["hp"].shape == (N_AGENTS,)

    def test_info_contains_positions(self, default_env):
        _, _, _, _, info = default_env.step([STAY, STAY])
        assert "positions" in info
        assert info["positions"].shape == (N_AGENTS, 2)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestArenaPerformance:
    def test_step_under_1ms(self):
        import time

        env = ArenaEnv()
        env.reset(seed=0)
        times = []
        n = 2000
        for k in range(n):
            t0 = time.perf_counter()
            env.step([STAY, STAY])
            times.append(time.perf_counter() - t0)
            if env._step_count >= env.max_steps:
                env.reset()

        avg_ms = float(np.mean(times)) * 1000.0
        assert avg_ms < 1.0, f"Average step time {avg_ms:.4f}ms exceeds 1ms budget"
