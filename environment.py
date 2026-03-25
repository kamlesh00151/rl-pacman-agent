"""
environment.py
==============
Defines the PacmanEnv class — a grid-based Pacman environment.

Grid legend:
  0 = empty cell
  1 = wall
  2 = food pellet
  3 = Pacman
  4 = ghost
"""

import numpy as np
import random

# ── Default maze layout (10 × 10) ──────────────────────────────────────────
# 1 = wall, 2 = food, 0 = empty
DEFAULT_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
    [1, 2, 1, 2, 1, 2, 1, 1, 2, 1],
    [1, 2, 1, 2, 2, 2, 2, 1, 2, 1],
    [1, 2, 1, 1, 1, 1, 2, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
    [1, 2, 2, 1, 2, 2, 2, 1, 2, 1],
    [1, 2, 2, 2, 2, 1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Action constants
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
NUM_ACTIONS  = 4

# Direction vectors (row_delta, col_delta)
DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
}


class Ghost:
    """Simple ghost that moves randomly (avoids walls)."""

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def move(self, grid: np.ndarray):
        """Move to a random adjacent non-wall cell."""
        rows, cols = grid.shape
        options = []
        for dr, dc in DELTAS.values():
            nr, nc = self.row + dr, self.col + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != 1:
                options.append((nr, nc))
        if options:
            self.row, self.col = random.choice(options)


class PacmanEnv:
    """
    Grid-based Pacman environment compatible with a gym-style interface.

    Parameters
    ----------
    maze_template : list[list[int]] | None
        Custom maze layout. Uses DEFAULT_MAZE when None.
    num_ghosts : int
        Number of ghosts to spawn (1 or 2).
    """

    def __init__(self, maze_template=None, num_ghosts: int = 2):
        self.maze_template = np.array(
            maze_template if maze_template else DEFAULT_MAZE, dtype=np.int32
        )
        self.num_ghosts = num_ghosts
        self.rows, self.cols = self.maze_template.shape

        # Pre-compute all non-wall positions for random spawning
        self._open_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.maze_template[r, c] != 1
        ]

        # Compute total food count once
        self.total_food = int((self.maze_template == 2).sum())

        # State size: pacman(2) + ghost(2*num_ghosts) + flattened food grid
        self.state_size = 2 + 2 * self.num_ghosts + self.rows * self.cols

        self.reset()

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset environment to initial state; return initial observation."""
        # Working copy of maze (food positions may change mid-episode)
        self.grid = self.maze_template.copy()

        # Place Pacman at a random open cell that has food
        food_cells = [(r, c) for r, c in self._open_cells if self.grid[r, c] == 2]
        self.pac_row, self.pac_col = random.choice(food_cells)

        # Place ghosts far from Pacman
        ghost_candidates = [
            pos for pos in self._open_cells
            if abs(pos[0] - self.pac_row) + abs(pos[1] - self.pac_col) > 4
        ]
        if len(ghost_candidates) < self.num_ghosts:
            ghost_candidates = self._open_cells  # fallback

        ghost_starts = random.sample(ghost_candidates, self.num_ghosts)
        self.ghosts = [Ghost(r, c) for r, c in ghost_starts]

        # Counters
        self.food_eaten  = 0
        self.steps_taken = 0
        self.done        = False
        self.score       = 0

        return self._get_state()

    def step(self, action: int):
        """
        Apply action, advance ghosts, compute reward.

        Returns
        -------
        state  : np.ndarray
        reward : float
        done   : bool
        info   : dict
        """
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")

        reward = -1  # step penalty
        self.steps_taken += 1

        # ── Move Pacman ──────────────────────────────────────────────────
        dr, dc = DELTAS[action]
        nr, nc = self.pac_row + dr, self.pac_col + dc

        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] != 1:
            self.pac_row, self.pac_col = nr, nc

        # Check food
        if self.grid[self.pac_row, self.pac_col] == 2:
            self.grid[self.pac_row, self.pac_col] = 0  # consume food
            self.food_eaten += 1
            reward += 10
            self.score += 10

        # Check all food cleared
        if self.food_eaten == self.total_food:
            reward += 50
            self.score += 50
            self.done = True

        # ── Move Ghosts ──────────────────────────────────────────────────
        if not self.done:
            for ghost in self.ghosts:
                ghost.move(self.grid)

        # ── Check collision ───────────────────────────────────────────────
        if not self.done and self._caught_by_ghost():
            reward = -100
            self.score -= 100
            self.done = True

        state = self._get_state()
        info  = {"score": self.score, "food_eaten": self.food_eaten, "steps": self.steps_taken}
        return state, reward, self.done, info

    def render_text(self):
        """Print a simple ASCII representation of the current state."""
        display = self.grid.copy()
        display[self.pac_row, self.pac_col] = 3
        for ghost in self.ghosts:
            display[ghost.row, ghost.col] = 4
        symbols = {0: "  ", 1: "██", 2: "·", 3: "C", 4: "G"}
        for row in display:
            print("".join(symbols.get(cell, "?") for cell in row))
        print(f"Score: {self.score}  Food: {self.food_eaten}/{self.total_food}")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _caught_by_ghost(self) -> bool:
        return any(g.row == self.pac_row and g.col == self.pac_col for g in self.ghosts)

    def _get_state(self) -> np.ndarray:
        """
        Build the state vector:
          [pac_row/rows, pac_col/cols,
           ghost0_row/rows, ghost0_col/cols, (ghost1…),
           flattened_food_grid (0/1 per cell)]
        All values normalised to [0, 1].
        """
        parts = [
            self.pac_row / self.rows,
            self.pac_col / self.cols,
        ]
        for ghost in self.ghosts:
            parts.append(ghost.row / self.rows)
            parts.append(ghost.col / self.cols)
        # Pad missing ghosts (if num_ghosts < max)
        for _ in range(self.num_ghosts - len(self.ghosts)):
            parts.extend([0.0, 0.0])

        food_map = (self.grid == 2).astype(np.float32).flatten()
        return np.array(parts + list(food_map), dtype=np.float32)
