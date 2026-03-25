"""
train.py
========
Training loop for the RL Pacman agent.

Usage
-----
  python train.py                   # train with default settings
  python train.py --episodes 1000   # custom episode count
  python train.py --headless        # no pygame window during training (faster)
"""

import argparse
import os
import numpy as np

from environment import PacmanEnv, NUM_ACTIONS
from agent      import DQNAgent
from utils      import plot_training_curves, print_progress, print_summary, ensure_dir


# ── Default hyper-parameters ─────────────────────────────────────────────────
DEFAULTS = dict(
    episodes      = 800,
    max_steps     = 300,        # max steps per episode before forced termination
    num_ghosts    = 2,
    lr            = 1e-3,
    gamma         = 0.95,
    epsilon_start = 1.0,
    epsilon_end   = 0.05,
    epsilon_decay = 0.003,      # decrease per episode
    batch_size    = 64,
    target_update = 10,
    buffer_cap    = 10_000,
    learn_freq    = 4,          # learn every N environment steps
    model_dir     = "models",
    model_name    = "pacman_dqn.pth",
    curves_path   = "training_curves.png",
    print_every   = 50,
)


def train(cfg: dict):
    """Run the full training loop."""
    ensure_dir(cfg["model_dir"])

    # ── Build environment & agent ──────────────────────────────────────────
    env = PacmanEnv(num_ghosts=cfg["num_ghosts"])

    agent = DQNAgent(
        state_size    = env.state_size,
        num_actions   = NUM_ACTIONS,
        lr            = cfg["lr"],
        gamma         = cfg["gamma"],
        epsilon_start = cfg["epsilon_start"],
        epsilon_end   = cfg["epsilon_end"],
        epsilon_decay = cfg["epsilon_decay"],
        batch_size    = cfg["batch_size"],
        target_update = cfg["target_update"],
        buffer_cap    = cfg["buffer_cap"],
    )

    # ── History buffers ────────────────────────────────────────────────────
    ep_rewards  = []
    ep_lengths  = []
    ep_epsilon  = []
    ep_food_pct = []

    print("\n" + "═" * 52)
    print("   RL Pacman Mini Agent — Training Started")
    print("═" * 52)
    print(f"  State size  : {env.state_size}")
    print(f"  Episodes    : {cfg['episodes']}")
    print(f"  Max steps   : {cfg['max_steps']}")
    print(f"  Ghosts      : {cfg['num_ghosts']}")
    print(f"  Device      : {agent.device}")
    print("═" * 52 + "\n")

    global_step = 0

    for ep in range(1, cfg["episodes"] + 1):
        state    = env.reset()
        ep_reward = 0.0

        for step in range(cfg["max_steps"]):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.buffer.push(state, action, reward, next_state, done)

            state      = next_state
            ep_reward += reward
            global_step += 1

            # Learn from replay buffer every `learn_freq` steps
            if global_step % cfg["learn_freq"] == 0:
                agent.learn()

            if done:
                break

        # ── End of episode ─────────────────────────────────────────────
        agent.end_episode()

        food_pct = info["food_eaten"] / env.total_food

        ep_rewards.append(ep_reward)
        ep_lengths.append(step + 1)
        ep_epsilon.append(agent.epsilon)
        ep_food_pct.append(food_pct)

        # Rolling average over last 50 episodes
        avg_r = float(np.mean(ep_rewards[-50:]))

        # Console progress
        print_progress(ep, cfg["episodes"], ep_reward, agent.epsilon, avg_r, food_pct)

        if ep % cfg["print_every"] == 0:
            print()  # newline after progress bar
            print(
                f"  [Ep {ep:>4}] Avg-50: {avg_r:>7.1f} | "
                f"Food: {food_pct*100:>5.1f}% | "
                f"Buffer: {len(agent.buffer)}"
            )

    print()  # final newline

    # ── Save model & plots ─────────────────────────────────────────────────
    model_path = os.path.join(cfg["model_dir"], cfg["model_name"])
    agent.save(model_path)

    plot_training_curves(ep_rewards, ep_lengths, ep_epsilon, cfg["curves_path"])
    print_summary(ep_rewards, ep_lengths)

    return agent, ep_rewards


# ── CLI entry-point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train RL Pacman DQN agent")
    p.add_argument("--episodes",       type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--max_steps",      type=int,   default=DEFAULTS["max_steps"])
    p.add_argument("--num_ghosts",     type=int,   default=DEFAULTS["num_ghosts"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--epsilon_decay",  type=float, default=DEFAULTS["epsilon_decay"])
    p.add_argument("--model_dir",      type=str,   default=DEFAULTS["model_dir"])
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = {**DEFAULTS, **parse_args()}
    train(cfg)
