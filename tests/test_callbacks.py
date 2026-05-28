"""Unit and integration tests for utils/callbacks.py"""

import pytest
from env.reward_shaping import RewardShaper
from utils.callbacks import CurriculumCallback
from training.self_play_loop import run_self_play


@pytest.fixture
def shaper():
    return RewardShaper(target_zone=(5.0, 5.0), initial_weight=1.0)


# ---------------------------------------------------------------------------
# CurriculumCallback — weight decay formula
# ---------------------------------------------------------------------------

class TestCurriculumCallbackWeightDecay:
    def _make(self, shaper, total=1000, initial=1.0, final=0.0):
        cb = CurriculumCallback(shaper, total_timesteps=total,
                                initial_weight=initial, final_weight=final)
        return cb

    def test_weight_at_zero_timesteps(self, shaper):
        cb = self._make(shaper)
        cb.num_timesteps = 0
        cb._on_step()
        assert shaper.weight == pytest.approx(1.0)

    def test_weight_at_halfway(self, shaper):
        cb = self._make(shaper)
        cb.num_timesteps = 500
        cb._on_step()
        assert shaper.weight == pytest.approx(0.5)

    def test_weight_at_completion(self, shaper):
        cb = self._make(shaper)
        cb.num_timesteps = 1000
        cb._on_step()
        assert shaper.weight == pytest.approx(0.0)

    def test_weight_clamped_past_total(self, shaper):
        cb = self._make(shaper)
        cb.num_timesteps = 99_999
        cb._on_step()
        assert shaper.weight == pytest.approx(0.0)

    def test_non_zero_final_weight(self, shaper):
        cb = self._make(shaper, total=100, initial=1.0, final=0.3)
        cb.num_timesteps = 100
        cb._on_step()
        assert shaper.weight == pytest.approx(0.3)

    def test_weight_monotone_decreasing(self, shaper):
        cb = self._make(shaper)
        weights = []
        for t in range(0, 1100, 100):
            cb.num_timesteps = t
            cb._on_step()
            weights.append(shaper.weight)
        assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))

    def test_on_step_returns_true(self, shaper):
        cb = self._make(shaper)
        cb.num_timesteps = 50
        assert cb._on_step() is True

    def test_shaper_is_updated_not_replaced(self, shaper):
        original_id = id(shaper)
        cb = self._make(shaper)
        cb.num_timesteps = 500
        cb._on_step()
        assert id(shaper) == original_id  # same object, mutated in-place


# ---------------------------------------------------------------------------
# Integration: CurriculumCallback + run_self_play
# ---------------------------------------------------------------------------

class TestCurriculumIntegration:
    _FAST_PPO = dict(n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    _FAST_ARENA = dict(max_steps=20)

    def test_callback_decays_weight_during_training(self, tmp_path):
        shaper = RewardShaper(target_zone=(5.0, 5.0), initial_weight=1.0)
        callback = CurriculumCallback(shaper, total_timesteps=128,
                                      initial_weight=1.0, final_weight=0.0)
        run_self_play(
            total_timesteps=128,
            generation_steps=64,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=None,
            arena_kwargs={**self._FAST_ARENA, "shaper": shaper},
            ppo_kwargs=self._FAST_PPO,
            callbacks=[callback],
        )
        # Weight should have decayed toward final_weight
        assert shaper.weight < 1.0

    def test_run_with_shaper_and_callback_completes(self, tmp_path):
        shaper = RewardShaper(target_zone=(5.0, 5.0), initial_weight=0.5)
        callback = CurriculumCallback(shaper, total_timesteps=64)
        run_self_play(
            total_timesteps=64,
            generation_steps=64,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=None,
            arena_kwargs={**self._FAST_ARENA, "shaper": shaper},
            ppo_kwargs=self._FAST_PPO,
            callbacks=[callback],
        )
