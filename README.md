# NeuralWorld v0.2 — State World Model

NeuralWorld is an experimental neural game engine where an AI will learn to
imagine the future, act on it, and eventually turn that imagination into music.

Version 0.2 adds the first learned **state world model** to the deterministic
2D box-pushing environment. A PyTorch MLP learns the one-step transition
`(state_t, action_t) → state_t+1` from reproducibly generated gameplay data.

## What is included

- Agent, walls, a pushable box, and a goal
- Deterministic grid physics and collision handling
- Step, collision, push, and goal rewards
- Gymnasium-compatible `reset` and `step` API
- Seeded layout selection and reproducible trajectories
- Headless state and `rgb_array` operation (Pygame is not initialized)
- Optional keyboard-controlled Pygame window
- Unit tests for physics, API compliance, rendering, and reproducibility
- Episode-level train / validation / test dataset splitting
- Configurable PyTorch MLP for next-state prediction
- Held-out one-step MSE, coordinate accuracy, exact-state accuracy, and grid MAE
- Reusable model checkpoints and JSON experiment metrics

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

For headless servers that do not need the interactive window, install only the
core environment:

```bash
python -m pip install -e ".[dev]" --no-deps
python -m pip install "gymnasium>=1.0,<2" "numpy>=1.24,<3" "torch>=2.3,<3" "pytest>=8,<9"
```

## Quick start

```python
from game import Action, NeuralWorldEnv

env = NeuralWorldEnv(render_mode=None)
observation, info = env.reset(seed=42)

for action in [Action.RIGHT, Action.RIGHT, Action.DOWN]:
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

Observations expose the ground-truth state directly:

```text
player     int32[2]          (x, y)
box        int32[2]          (x, y)
goal       int32[2]          (x, y)
walls      int8[height,width]
step_count integer
```

Actions are discrete: `NOOP`, `UP`, `DOWN`, `LEFT`, and `RIGHT`.

## Play

Install the default dependencies, then run:

```bash
python -m game.play --seed 42
```

Use the arrow keys or WASD to move, `R` to reset, and Escape to quit. Reaching
the green goal ends the episode; the orange box can be pushed into free cells.

## Headless trajectories

The environment never imports Pygame unless `render_mode="human"` is selected.
This makes dataset generation safe on servers and CI:

```python
env = NeuralWorldEnv(render_mode="rgb_array")
obs, _ = env.reset(seed=7)
frame = env.render()  # uint8 NumPy array, shape (288, 288, 3)
```

Given the same seed and action sequence, observations, rewards, termination
flags, and event information are identical. This is covered by the test suite.

## State world model

Generate a gameplay dataset, train the MLP, and evaluate it on the held-out test
episodes with one command:

```bash
python -m experiments.one_step --regenerate
```

The reproducible experiment settings live in `configs/state_mlp.toml`. Outputs
are written beneath `artifacts/v0.2/` and are intentionally ignored by Git:

```text
transitions.npz  encoded transitions and episode-level split labels
state_mlp.pt     best-validation PyTorch checkpoint
metrics.json     training summary and held-out test metrics
```

To generate data separately:

```bash
python -m data.generate --output artifacts/transitions.npz --episodes 300 --steps 100 --seed 42
```

The input vector contains normalized Agent, Box, and Goal coordinates plus the
binary wall grid. The action is one-hot encoded and concatenated inside the
model. The prediction target is the next normalized Agent and Box coordinates;
static Goal and Wall state is carried forward unchanged by the environment.

Episodes—not individual rows—are assigned to the train, validation, and test
splits, preventing adjacent transitions from the same trajectory from leaking
across splits. The test split is used only after selecting the best validation
checkpoint.

### Reference result

The default seed-42 CPU run produces 27,811 transitions, including 4,191 held-out
test transitions:

| Test metric | Result |
| --- | ---: |
| Normalized MSE | 0.000337 |
| Per-coordinate accuracy | 98.03% |
| Agent position accuracy | 93.20% |
| Box position accuracy | 98.90% |
| Exact next-state accuracy | 92.10% |
| Grid-cell MAE | 0.0197 |

Small floating-point differences may occur across PyTorch versions and hardware.

## Rewards

| Event | Reward |
| --- | ---: |
| Each step | -0.01 |
| Wall / blocked-box collision | additional -0.04 |
| Successful box push | additional +0.02 |
| Reach goal | additional +1.00 |

Episodes terminate at the goal and truncate after 200 steps by default.

## Development

```bash
pytest
```

Repository layout:

```text
game/                   ground-truth environment and renderer
data/                   transition generation, encoding, and split loading
models/state_mlp.py     one-step dynamics baseline
experiments/one_step.py reproducible training and held-out evaluation
configs/                environment and model experiment parameters
tests/                  engine, dataset, and model tests
```

## Roadmap

- **v0.1:** deterministic ground-truth world
- **v0.2:** state world model (this release)
- **v0.3:** long-horizon learned rollout
- **v0.4:** model-based planning
- **v0.5:** theory-based adaptive music
