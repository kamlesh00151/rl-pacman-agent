"""
main.py
=======
Launches the Pygame visualisation of the trained (or random) Pacman agent.

Modes
-----
  python main.py --mode play        # watch the trained agent play
  python main.py --mode train_play  # train first, then watch
  python main.py --mode random      # watch a random agent (sanity check)

Controls (during play)
------
  Q / ESC  — quit
  SPACE    — pause / resume
  R        — restart episode
"""

import sys
import os
import argparse
import time

import numpy as np
import pygame

from environment import PacmanEnv, NUM_ACTIONS, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT
from agent       import DQNAgent
from train       import train, DEFAULTS


# ── Colours ──────────────────────────────────────────────────────────────────
BLACK       = (  0,   0,   0)
DARK_BLUE   = (  0,   0,  40)
WALL_COLOR  = ( 30,  30, 150)
FOOD_COLOR  = (255, 200,   0)
PAC_COLOR   = (255, 220,   0)
GHOST_COLORS = [
    (255,  80,  80),   # red ghost
    ( 80, 200, 255),   # blue ghost
]
TEXT_COLOR  = (255, 255, 255)
BG_COLOR    = (  5,   5,  20)

# ── Layout ────────────────────────────────────────────────────────────────────
CELL_SIZE   = 52     # pixels per grid cell
HUD_HEIGHT  = 70     # pixels reserved for the top HUD bar
FPS         = 8      # frames per second (slowed so it's watchable)


class PacmanRenderer:
    """Handles all Pygame drawing for the Pacman environment."""

    def __init__(self, env: PacmanEnv):
        self.env   = env
        self.rows  = env.rows
        self.cols  = env.cols

        width  = self.cols * CELL_SIZE
        height = self.rows * CELL_SIZE + HUD_HEIGHT

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("RL Pacman Mini Agent")
        self.clock = pygame.time.Clock()

        self.font_big   = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14)

    def draw(self, episode: int, step: int, score: int,
             food_eaten: int, total_food: int, epsilon: float, paused: bool):
        """Render one frame."""
        self.screen.fill(BG_COLOR)
        self._draw_hud(episode, step, score, food_eaten, total_food, epsilon, paused)
        self._draw_grid()
        self._draw_entities()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_hud(self, episode, step, score, food_eaten, total_food, epsilon, paused):
        """Top information bar."""
        pygame.draw.rect(self.screen, (10, 10, 50), (0, 0, self.cols * CELL_SIZE, HUD_HEIGHT))
        pygame.draw.line(self.screen, WALL_COLOR,
                         (0, HUD_HEIGHT - 1), (self.cols * CELL_SIZE, HUD_HEIGHT - 1), 2)

        texts = [
            f"EPISODE {episode}",
            f"SCORE {score:>6}",
            f"FOOD {food_eaten}/{total_food}",
            f"STEP {step:>4}",
            f"ε={epsilon:.3f}",
        ]
        if paused:
            texts.append("⏸  PAUSED")

        x = 8
        for t in texts:
            surf = self.font_big.render(t, True, TEXT_COLOR)
            self.screen.blit(surf, (x, 22))
            x += surf.get_width() + 20

    def _draw_grid(self):
        """Draw walls and food pellets."""
        env = self.env
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE + HUD_HEIGHT, CELL_SIZE, CELL_SIZE)
                cell = env.grid[r, c]

                if cell == 1:  # wall
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)
                    # Inner highlight for 3-D look
                    highlight = pygame.Rect(rect.x + 2, rect.y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                    pygame.draw.rect(self.screen, (50, 50, 180), highlight)
                elif cell == 2:  # food pellet
                    cx = c * CELL_SIZE + CELL_SIZE // 2
                    cy = r * CELL_SIZE + CELL_SIZE // 2 + HUD_HEIGHT
                    pygame.draw.circle(self.screen, FOOD_COLOR, (cx, cy), 5)
                else:            # empty
                    pygame.draw.rect(self.screen, DARK_BLUE, rect, 1)

    def _draw_entities(self):
        """Draw Pacman and ghosts."""
        env = self.env

        # Pacman (yellow circle with mouth)
        cx = env.pac_col * CELL_SIZE + CELL_SIZE // 2
        cy = env.pac_row * CELL_SIZE + CELL_SIZE // 2 + HUD_HEIGHT
        pygame.draw.circle(self.screen, PAC_COLOR, (cx, cy), CELL_SIZE // 2 - 5)
        # Simple mouth cut-out
        pygame.draw.polygon(
            self.screen, DARK_BLUE,
            [(cx, cy),
             (cx + CELL_SIZE // 2 - 5, cy - 8),
             (cx + CELL_SIZE // 2 - 5, cy + 8)],
        )

        # Ghosts
        for idx, ghost in enumerate(env.ghosts):
            color = GHOST_COLORS[idx % len(GHOST_COLORS)]
            gx = ghost.col * CELL_SIZE + CELL_SIZE // 2
            gy = ghost.row * CELL_SIZE + CELL_SIZE // 2 + HUD_HEIGHT
            r  = CELL_SIZE // 2 - 5
            # Body
            pygame.draw.circle(self.screen, color, (gx, gy), r)
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(gx - r, gy, r * 2, r))
            # Eyes (white dots)
            eye_r = max(3, r // 4)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx - r // 3, gy - r // 4), eye_r)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx + r // 3, gy - r // 4), eye_r)
            # Pupils
            pygame.draw.circle(self.screen, (0, 0, 200), (gx - r // 3, gy - r // 4), max(1, eye_r - 1))
            pygame.draw.circle(self.screen, (0, 0, 200), (gx + r // 3, gy - r // 4), max(1, eye_r - 1))

    def quit(self):
        pygame.quit()


# ── Play loop ─────────────────────────────────────────────────────────────────

def play_loop(env: PacmanEnv, agent: DQNAgent, episodes: int = 10, epsilon: float = 0.05):
    """Watch the agent play for `episodes` games with Pygame rendering."""
    renderer = PacmanRenderer(env)
    agent.epsilon = epsilon   # low exploration during demo

    episode = 0
    while episode < episodes:
        state  = env.reset()
        done   = False
        step   = 0
        paused = False

        while not done:
            # ── Event handling ──────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        renderer.quit()
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_r:
                        done = True  # force restart

            if paused:
                renderer.draw(episode + 1, step, env.score,
                               env.food_eaten, env.total_food, agent.epsilon, paused)
                continue

            # ── Agent acts ──────────────────────────────────────────────
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)
            step += 1

            renderer.draw(episode + 1, step, env.score,
                           env.food_eaten, env.total_food, agent.epsilon, paused)

        episode += 1
        # Brief pause between episodes so the final frame is visible
        time.sleep(0.8)

    renderer.quit()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RL Pacman Mini Agent")
    p.add_argument("--mode",        choices=["play", "train_play", "random"],
                   default="train_play",
                   help="'train_play': train then watch | 'play': load model and watch | 'random': random agent")
    p.add_argument("--episodes",    type=int, default=DEFAULTS["episodes"],
                   help="Training episodes (used when mode=train_play)")
    p.add_argument("--play_eps",    type=int, default=10,
                   help="Number of games to visualise after training")
    p.add_argument("--model_path",  type=str,
                   default=os.path.join(DEFAULTS["model_dir"], DEFAULTS["model_name"]),
                   help="Path to .pth file (used when mode=play)")
    p.add_argument("--num_ghosts",  type=int, default=DEFAULTS["num_ghosts"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env  = PacmanEnv(num_ghosts=args.num_ghosts)

    agent = DQNAgent(
        state_size  = env.state_size,
        num_actions = NUM_ACTIONS,
    )

    if args.mode == "train_play":
        # Train first
        cfg = {**DEFAULTS, "episodes": args.episodes, "num_ghosts": args.num_ghosts}
        agent, _ = train(cfg)
        print("\nTraining done! Launching gameplay demo…")
        play_loop(env, agent, episodes=args.play_eps)

    elif args.mode == "play":
        if not os.path.exists(args.model_path):
            print(f"[Error] Model not found at '{args.model_path}'.")
            print("  Run 'python main.py --mode train_play' first.")
            sys.exit(1)
        agent.load(args.model_path)
        play_loop(env, agent, episodes=args.play_eps)

    elif args.mode == "random":
        # Random agent — useful for baseline / sanity check
        class RandomAgent:
            epsilon = 1.0
            def select_action(self, state, training=True):
                import random
                return random.randrange(NUM_ACTIONS)
        play_loop(env, RandomAgent(), episodes=args.play_eps)
