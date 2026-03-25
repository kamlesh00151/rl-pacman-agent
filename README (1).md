# 🟡 RL Pacman Mini Agent

A Deep Q-Network (DQN) agent that learns to play a simplified Pacman game — collecting food pellets while dodging ghosts — built as an AI/ML college mini project.

---

## 📁 Project Structure

```
rl_pacman/
├── environment.py   # Grid-based Pacman world (maze, ghosts, rewards)
├── agent.py         # DQN agent (neural network + replay buffer + target network)
├── train.py         # Training loop (epsilon-greedy, periodic saving)
├── main.py          # Pygame visualisation + entry-point
├── utils.py         # Plotting, progress display, summary stats
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Train AND watch the agent
```bash
python main.py --mode train_play --episodes 800
```

### 3 · Watch a saved model (after training)
```bash
python main.py --mode play
```

### 4 · Watch a random agent (baseline)
```bash
python main.py --mode random
```

### 5 · Train only (headless, no Pygame)
```bash
python train.py --episodes 1000
```

---

## 🎮 Controls (Pygame window)

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `R` | Restart episode |
| `Q` / `ESC` | Quit |

---

## 🧠 Algorithm — Deep Q-Network (DQN)

| Component | Detail |
|-----------|--------|
| Network | 3-layer MLP: 128 → 128 → 64 |
| Optimiser | Adam (lr = 1e-3) |
| Discount γ | 0.95 |
| Replay buffer | 10 000 transitions |
| Target network sync | Every 10 episodes |
| Exploration | ε-greedy, 1.0 → 0.05 (linear decay) |
| Batch size | 64 |

### State vector
`[pac_row, pac_col, ghost0_row, ghost0_col, ghost1_row, ghost1_col, food_map…]`  
All values normalised to **[0, 1]**.

### Reward shaping

| Event | Reward |
|-------|--------|
| Eat food pellet | **+10** |
| Clear all food | **+50** |
| Caught by ghost | **−100** |
| Each step | **−1** |

---

## ⚙️ Hyper-parameter Reference (`train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodes` | 800 | Total training episodes |
| `max_steps` | 300 | Max steps before episode ends |
| `num_ghosts` | 2 | Number of ghosts |
| `lr` | 1e-3 | Learning rate |
| `gamma` | 0.95 | Discount factor |
| `epsilon_decay` | 0.003 | ε decrease per episode |
| `batch_size` | 64 | Replay batch size |
| `target_update` | 10 | Target-net sync interval |
| `learn_freq` | 4 | Learn every N environment steps |

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `models/pacman_dqn.pth` | Saved model weights |
| `training_curves.png` | Reward / length / ε plots |

---

## 🗺️ Maze Legend

```
██  Wall
·   Food pellet
C   Pacman (agent)
G   Ghost
```

---

## 💡 Extension Ideas

- Larger / randomly-generated mazes
- Power pellets (agent can eat ghosts temporarily)
- Multiple difficulty levels (ghost speed, number of ghosts)
- Convolutional DQN using the raw grid image as state
- Prioritised experience replay

---

*Built with Python · NumPy · PyTorch · Pygame*
