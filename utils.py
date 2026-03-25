"""
utils.py
========
Helper utilities:
  - Moving-average calculator
  - Plotting training curves
  - Printing a formatted training summary table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt


def moving_average(values: list, window: int = 20) -> np.ndarray:
    """Return a smoothed version of `values` using a simple moving average."""
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(
    episode_rewards: list,
    episode_lengths: list,
    epsilon_history: list,
    save_path: str = "training_curves.png",
):
    """
    Save a 3-panel figure showing:
      1. Episode rewards + smoothed trend
      2. Episode lengths (steps per episode)
      3. Epsilon decay over episodes
    """
    episodes = range(1, len(episode_rewards) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("RL Pacman — Training Curves", fontsize=14, fontweight="bold")

    # ── Panel 1: Rewards ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color="royalblue", label="Raw reward")
    smoothed = moving_average(episode_rewards, window=20)
    ax.plot(
        range(len(episode_rewards) - len(smoothed) + 1, len(episode_rewards) + 1),
        smoothed,
        color="royalblue",
        linewidth=2,
        label="MA-20",
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Episode length ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(episodes, episode_lengths, alpha=0.4, color="darkorange", label="Steps")
    smoothed_len = moving_average(episode_lengths, window=20)
    ax.plot(
        range(len(episode_lengths) - len(smoothed_len) + 1, len(episode_lengths) + 1),
        smoothed_len,
        color="darkorange",
        linewidth=2,
        label="MA-20",
    )
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Epsilon decay ────────────────────────────────────────────
    ax = axes[2]
    ax.plot(episodes, epsilon_history, color="crimson", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Exploration Rate (ε-Greedy Decay)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[Utils] Training curves saved → {save_path}")


def print_progress(episode: int, total: int, reward: float,
                   epsilon: float, avg_reward: float, food_pct: float):
    """Print a one-line training progress update."""
    bar_len  = 20
    filled   = int(bar_len * episode / total)
    bar      = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r[{bar}] Ep {episode:>4}/{total}"
        f"  R={reward:>7.1f}"
        f"  Avg={avg_reward:>7.1f}"
        f"  ε={epsilon:.3f}"
        f"  Food%={food_pct*100:>5.1f}",
        end="",
        flush=True,
    )


def print_summary(episode_rewards: list, episode_lengths: list):
    """Print final training statistics."""
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    n = len(rewards)
    print("\n\n" + "=" * 52)
    print("          TRAINING COMPLETE — SUMMARY")
    print("=" * 52)
    print(f"  Total episodes      : {n}")
    print(f"  Mean reward         : {rewards.mean():.2f}")
    print(f"  Std reward          : {rewards.std():.2f}")
    print(f"  Max reward          : {rewards.max():.2f}")
    print(f"  Min reward          : {rewards.min():.2f}")
    print(f"  Mean episode length : {lengths.mean():.1f} steps")
    # Best 10 %
    top10 = np.percentile(rewards, 90)
    print(f"  Top-10% reward ≥    : {top10:.2f}")
    print("=" * 52 + "\n")


def ensure_dir(path: str):
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
