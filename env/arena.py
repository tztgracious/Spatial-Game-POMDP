"""
2-player FPS microstructure arena built on Gymnasium.

Both agents act simultaneously via a MultiDiscrete action space.
The environment is intentionally a single Gymnasium env (not PettingZoo)
so that stable-baselines3 self-play wrappers can be applied directly in
Phase 2 without changing the core step logic.

Observation vector (length 14, two 7-element blocks, one per agent):
    [own_x, own_y, own_hp_norm, los, opp_x, opp_y, opp_hp_norm]
    When LoS = False, opp_x / opp_y / opp_hp_norm are set to 0.0
    to enforce partial observability.

Action encoding (per agent):
    0 = stay  1 = up  2 = down  3 = left  4 = right  5 = fire
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.geometry import has_line_of_sight, Segment

# Action indices
STAY, UP, DOWN, LEFT, RIGHT, FIRE = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6

_MOVE_DELTA: Dict[int, Tuple[float, float]] = {
    UP: (0.0, 1.0),
    DOWN: (0.0, -1.0),
    LEFT: (-1.0, 0.0),
    RIGHT: (1.0, 0.0),
}

OBS_SIZE = 7  # per-agent observation block length
N_AGENTS = 2


class ArenaEnv(gym.Env):
    """
    Two-player zero-sum FPS arena with Line-of-Sight partial observability.

    Parameters
    ----------
    width, height : float
        Arena dimensions.
    walls : list of Segment, optional
        Wall segments that block LoS.  Default is a single vertical wall
        bisecting the centre of the arena to create a corner peeking scenario.
    step_size : float
        Distance each move action translates an agent per step.
    max_hp : float
        Starting HP for each agent.
    fire_damage : float
        HP deducted from the target per successful (LoS) fire action.
    blind_fire_penalty : float
        Extra reward penalty applied to the shooter when firing without LoS.
    time_penalty : float
        Small negative reward applied to both agents each step to incentivise
        decisive play.
    kill_reward : float
        Reward bonus/penalty on kill/death (magnitude; sign applied automatically).
    max_steps : int
        Episode length cap before truncation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        walls: Optional[List[Segment]] = None,
        step_size: float = 0.5,
        max_hp: float = 100.0,
        fire_damage: float = 25.0,
        blind_fire_penalty: float = 0.5,
        time_penalty: float = 0.01,
        kill_reward: float = 10.0,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.width = width
        self.height = height
        self.step_size = step_size
        self.max_hp = max_hp
        self.fire_damage = fire_damage
        self.blind_fire_penalty = blind_fire_penalty
        self.time_penalty = time_penalty
        self.kill_reward = kill_reward
        self.max_steps = max_steps
        self.render_mode = render_mode

        if walls is None:
            mid_x, mid_y = width / 2.0, height / 2.0
            self.walls: List[Segment] = [
                ((mid_x, mid_y - 2.0), (mid_x, mid_y + 2.0))
            ]
        else:
            self.walls = walls

        # --- Spaces ---
        obs_low = np.zeros(N_AGENTS * OBS_SIZE, dtype=np.float32)
        obs_high = np.array(
            [width, height, 1.0, 1.0, width, height, 1.0] * N_AGENTS,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])

        # --- Mutable state (initialised in reset) ---
        self._positions: np.ndarray = np.zeros((N_AGENTS, 2), dtype=np.float32)
        self._hp: np.ndarray = np.zeros(N_AGENTS, dtype=np.float32)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._positions[0] = [1.0, 1.0]
        self._positions[1] = [self.width - 1.0, self.height - 1.0]
        self._hp[:] = self.max_hp
        self._step_count = 0

        return self._compute_obs(), {}

    def step(
        self, action: np.ndarray | List[int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = list(action)
        assert len(action) == N_AGENTS, f"Expected {N_AGENTS} actions, got {len(action)}"

        self._step_count += 1
        rewards = np.zeros(N_AGENTS, dtype=np.float32)

        # --- Movement (both agents move before resolving combat) ---
        for i in range(N_AGENTS):
            dx, dy = _MOVE_DELTA.get(action[i], (0.0, 0.0))
            self._positions[i][0] = float(
                np.clip(self._positions[i][0] + dx * self.step_size, 0.0, self.width)
            )
            self._positions[i][1] = float(
                np.clip(self._positions[i][1] + dy * self.step_size, 0.0, self.height)
            )

        # --- LoS query (computed once, reused for obs and combat) ---
        los: bool = has_line_of_sight(
            tuple(self._positions[0]), tuple(self._positions[1]), self.walls
        )

        # --- Combat ---
        for i in range(N_AGENTS):
            if action[i] == FIRE:
                if los:
                    j = 1 - i
                    self._hp[j] = max(0.0, self._hp[j] - self.fire_damage)
                else:
                    rewards[i] -= self.blind_fire_penalty

        # --- Time penalty ---
        rewards -= self.time_penalty

        # --- Termination check ---
        terminated = False
        for i in range(N_AGENTS):
            if self._hp[i] <= 0.0:
                j = 1 - i
                rewards[j] += self.kill_reward
                rewards[i] -= self.kill_reward
                terminated = True

        truncated = self._step_count >= self.max_steps
        obs = self._compute_obs()

        info: Dict[str, Any] = {
            "los": los,
            "hp": self._hp.copy(),
            "positions": self._positions.copy(),
            "step": self._step_count,
            "rewards_both": rewards.copy(),
        }

        # Primary reward is agent-0's perspective (SB3 single-agent convention).
        return obs, float(rewards[0]), terminated, truncated, info

    def render(self) -> Optional[str]:
        grid_w, grid_h = 20, 20
        grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

        # Walls
        for (wx1, wy1), (wx2, wy2) in self.walls:
            for t in np.linspace(0.0, 1.0, 20):
                gx = int((wx1 + t * (wx2 - wx1)) / self.width * (grid_w - 1))
                gy = int((wy1 + t * (wy2 - wy1)) / self.height * (grid_h - 1))
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    grid[grid_h - 1 - gy][gx] = "#"

        # Agents
        for i, label in enumerate(["A", "B"]):
            gx = int(np.clip(self._positions[i][0] / self.width * (grid_w - 1), 0, grid_w - 1))
            gy = int(np.clip(self._positions[i][1] / self.height * (grid_h - 1), 0, grid_h - 1))
            grid[grid_h - 1 - gy][gx] = label

        los = has_line_of_sight(
            tuple(self._positions[0]), tuple(self._positions[1]), self.walls
        )
        lines = ["".join(row) for row in grid]
        lines.append(
            f"HP: A={self._hp[0]:.0f}  B={self._hp[1]:.0f}  "
            f"LoS={'Yes' if los else 'No '}  Step={self._step_count}"
        )
        output = "\n".join(lines)

        if self.render_mode == "ansi":
            return output
        print(output)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_obs(self) -> np.ndarray:
        los = has_line_of_sight(
            tuple(self._positions[0]), tuple(self._positions[1]), self.walls
        )
        los_f = float(los)

        blocks: List[float] = []
        for i in range(N_AGENTS):
            j = 1 - i
            own_x, own_y = float(self._positions[i][0]), float(self._positions[i][1])
            own_hp = self._hp[i] / self.max_hp
            if los:
                opp_x = float(self._positions[j][0])
                opp_y = float(self._positions[j][1])
                opp_hp = self._hp[j] / self.max_hp
            else:
                opp_x = opp_y = opp_hp = 0.0
            blocks.extend([own_x, own_y, own_hp, los_f, opp_x, opp_y, opp_hp])

        return np.array(blocks, dtype=np.float32)
