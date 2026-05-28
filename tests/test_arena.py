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
# Spawn randomisation
# ---------------------------------------------------------------------------

class TestArenaSpawnRandomisation:
    def test_fixed_spawn_uses_corners(self):
        env = ArenaEnv(randomize_spawn=False)
        env.reset(seed=0)
        np.testing.assert_array_almost_equal(env._positions[0], [1.0, 1.0])
        np.testing.assert_array_almost_equal(
            env._positions[1], [env.width - 1.0, env.height - 1.0]
        )

    def test_random_spawn_varies_across_seeds(self):
        env = ArenaEnv(randomize_spawn=True)
        env.reset(seed=0)
        pos_a = env._positions.copy()
        env.reset(seed=1)
        pos_b = env._positions.copy()
        assert not np.allclose(pos_a, pos_b)

    def test_random_spawn_same_seed_reproducible(self):
        env = ArenaEnv(randomize_spawn=True)
        env.reset(seed=42)
        pos_a = env._positions.copy()
        env.reset(seed=42)
        pos_b = env._positions.copy()
        np.testing.assert_array_almost_equal(pos_a, pos_b)

    def test_random_spawn_respects_min_separation(self):
        env = ArenaEnv(randomize_spawn=True)
        for s in range(20):
            env.reset(seed=s)
            d = np.linalg.norm(env._positions[0] - env._positions[1])
            assert d >= ArenaEnv._SPAWN_MIN_SEP - 1e-5

    def test_random_spawn_within_margin(self):
        env = ArenaEnv(randomize_spawn=True)
        for s in range(20):
            env.reset(seed=s)
            for i in range(N_AGENTS):
                x, y = env._positions[i]
                assert env._SPAWN_MARGIN <= x <= env.width - env._SPAWN_MARGIN
                assert env._SPAWN_MARGIN <= y <= env.height - env._SPAWN_MARGIN


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


# ---------------------------------------------------------------------------
# Phase 3: probabilistic combat
# ---------------------------------------------------------------------------

class TestProbabilisticCombat:
    """Statistical tests — run many fire actions and verify hit-rate distributions."""

    @staticmethod
    def _hit_rate(env, trials: int = 300) -> float:
        hits = 0
        for _ in range(trials):
            hp_before = env._hp[1]
            env.step([FIRE, STAY])
            if env._hp[1] < hp_before:
                hits += 1
            env._hp[1] = env.max_hp  # reset HP so episode never terminates
        return hits / trials

    def test_distance_scale_none_is_deterministic(self):
        """Default (distance_scale=None) should always hit — backwards-compatible."""
        env = ArenaEnv(walls=[], distance_scale=None)
        env.reset(seed=0)
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        rate = self._hit_rate(env)
        assert rate == pytest.approx(1.0)

    def test_hit_rate_decreases_with_distance(self):
        """Hit rate must be lower at long range than close range."""
        env_close = ArenaEnv(walls=[], distance_scale=5.0)
        env_close.reset(seed=0)
        env_close._positions[0] = [4.5, 5.0]
        env_close._positions[1] = [5.5, 5.0]   # dist = 1

        env_far = ArenaEnv(walls=[], distance_scale=5.0)
        env_far.reset(seed=0)
        env_far._positions[0] = [1.0, 5.0]
        env_far._positions[1] = [9.0, 5.0]     # dist = 8

        rate_close = self._hit_rate(env_close)
        rate_far = self._hit_rate(env_far)
        assert rate_close > rate_far

    def test_hit_rate_below_one_at_long_range(self):
        """At 8 units with distance_scale=5, expected rate ≈ exp(-1.6) ≈ 0.20."""
        env = ArenaEnv(walls=[], distance_scale=5.0)
        env.reset(seed=42)
        env._positions[0] = [1.0, 5.0]
        env._positions[1] = [9.0, 5.0]   # dist = 8
        rate = self._hit_rate(env, trials=400)
        assert rate < 0.5  # clearly below certain hit

    def test_corner_peek_reduces_hit_rate(self):
        """Target barely peeking around a corner is harder to hit than one fully exposed.

        Option C geometry: wall from (5,0)→(5,7), A=(2,8) above the wall.
        B_peek at (5.4, 7.2) has just cleared the upper corner — small d_peek → low exposure.
        B_open at (8.0, 7.5) is far past the corner — large d_peek → high exposure.
        """
        walls = [((5.0, 0.0), (5.0, 7.0))]

        env_peek = ArenaEnv(walls=walls, min_exposure=0.05)
        env_peek.reset(seed=0)
        env_peek._positions[0] = [2.0, 8.0]
        env_peek._positions[1] = [5.4, 7.2]   # barely past upper corner

        env_open = ArenaEnv(walls=walls, min_exposure=0.05)
        env_open.reset(seed=0)
        env_open._positions[0] = [2.0, 8.0]
        env_open._positions[1] = [8.0, 7.5]   # far from corner, fully exposed

        rate_peek = self._hit_rate(env_peek)
        rate_open = self._hit_rate(env_open)
        assert rate_peek < rate_open

    def test_hit_info_in_step_output(self):
        env = ArenaEnv(walls=[], distance_scale=5.0)
        env.reset(seed=0)
        env._positions[0] = [2.0, 5.0]
        env._positions[1] = [8.0, 5.0]
        _, _, _, _, info = env.step([FIRE, STAY])
        assert "hit_info" in info
        assert 0 in info["hit_info"]
        assert isinstance(info["hit_info"][0], bool)

    def test_no_damage_on_miss(self):
        """If hit_info says miss, HP must be unchanged."""
        env = ArenaEnv(walls=[], distance_scale=5.0)
        env.reset(seed=0)
        env._positions[0] = [1.0, 5.0]
        env._positions[1] = [9.0, 5.0]
        for _ in range(50):
            hp_before = env._hp[1]
            _, _, terminated, truncated, info = env.step([FIRE, STAY])
            if 0 in info["hit_info"] and not info["hit_info"][0]:
                assert env._hp[1] == hp_before
            env._hp[1] = env.max_hp
            if terminated or truncated:
                env.reset(seed=0)


# ---------------------------------------------------------------------------
# Phase 3: RewardShaper integration
# ---------------------------------------------------------------------------

class TestArenaShaperIntegration:
    def test_shaper_adds_positive_reward_when_approaching(self):
        from env.reward_shaping import RewardShaper
        shaper = RewardShaper(target_zone=(5.0, 5.0), initial_weight=10.0)
        env = ArenaEnv(walls=[], shaper=shaper)
        env.reset(seed=0)
        # Place agent 0 far left; moving right approaches target_zone
        env._positions[0] = [1.0, 5.0]
        env._positions[1] = [9.0, 5.0]
        _, reward, _, _, _ = env.step([RIGHT, STAY])  # agent 0 moves right
        # Shaping should push reward upward; net > time_penalty alone
        assert reward > -env.time_penalty - 1.0

    def test_shaper_none_gives_same_reward_as_baseline(self):
        env_base = ArenaEnv(walls=[])
        env_base.reset(seed=0)
        env_base._positions[0] = [5.0, 5.0]
        env_base._positions[1] = [5.0, 5.0]
        _, r_base, _, _, _ = env_base.step([STAY, STAY])

        env_shape = ArenaEnv(walls=[], shaper=None)
        env_shape.reset(seed=0)
        env_shape._positions[0] = [5.0, 5.0]
        env_shape._positions[1] = [5.0, 5.0]
        _, r_shape, _, _, _ = env_shape.step([STAY, STAY])

        assert r_base == pytest.approx(r_shape)

    def test_shaper_weight_zero_gives_same_reward_as_no_shaper(self):
        from env.reward_shaping import RewardShaper
        shaper = RewardShaper(target_zone=(5.0, 5.0), initial_weight=0.0)
        env_base = ArenaEnv(walls=[])
        env_base.reset(seed=0)
        env_base._positions[0] = [2.0, 5.0]
        env_base._positions[1] = [8.0, 5.0]
        _, r_base, _, _, _ = env_base.step([STAY, STAY])

        env_shaped = ArenaEnv(walls=[], shaper=shaper)
        env_shaped.reset(seed=0)
        env_shaped._positions[0] = [2.0, 5.0]
        env_shaped._positions[1] = [8.0, 5.0]
        _, r_shaped, _, _, _ = env_shaped.step([STAY, STAY])

        assert r_base == pytest.approx(r_shaped, abs=1e-5)
