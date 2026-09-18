# NeuralWorld v0.1 — Ground Truth World

NeuralWorld is an experimental neural game engine where an AI will learn to
imagine the future, act on it, and eventually turn that imagination into music.

Version 0.1 is the deterministic **ground-truth world**: a small 2D box-pushing
environment designed to become the reference simulator for later learned world
models. No AI is used at this stage.

## What is included

- Agent, walls, a pushable box, and a goal
- Deterministic grid physics and collision handling
- Step, collision, push, and goal rewards
- Gymnasium-compatible `reset` and `step` API
- Seeded layout selection and reproducible trajectories
- Headless state and `rgb_array` operation (Pygame is not initialized)
- Optional keyboard-controlled Pygame window
- Unit tests for physics, API compliance, rendering, and reproducibility

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
python -m pip install "gymnasium>=1.0,<2" "numpy>=1.24,<3" "pytest>=8,<9"
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
game/                 ground-truth environment and renderer
configs/default.yaml  documented v0.1 environment parameters
tests/                deterministic physics and API tests
```

## Roadmap

- **v0.1:** deterministic ground-truth world (this release)
- **v0.2:** state world model
- **v0.3:** long-horizon learned rollout
- **v0.4:** model-based planning
- **v0.5:** theory-based adaptive music
