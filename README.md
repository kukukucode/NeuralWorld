# NeuralWorld

ゲーム世界の変化をAIに学習させ、将来的に未来予測・行動計画・音楽表現へつなげる実験プロジェクトです。

v0.2 の2Dゲーム・State World Modelに加え、**v0.3-1（連続予測の基礎）**まで実装しています。音楽生成は今後追加する予定です。

## できること

- プレイヤー・壁・箱・ゴールのある2Dゲームをプレイ
- 同じseedと操作列から同じ結果を再現
- ゲームデータを生成し、PyTorchのMLPで次状態を学習・評価
- 操作列に沿って予測を繰り返す（Goal・壁は固定、Player・Boxを更新）

## セットアップ

Python 3.10以上が必要です。

```bash
python -m pip install -e ".[dev]"
```

## 実行

```bash
python -m game.play --seed 42              # ゲームをプレイ（矢印キー / WASD）
python -m experiments.one_step --regenerate # データ生成・学習・1-step評価
pytest                                     # テスト
```

学習設定は `configs/state_mlp.toml`、実験結果は `artifacts/v0.2/` に保存されます。

連続予測はPythonから使えます（先に上記の学習コマンドを実行）。

```python
import torch
from game import Action, NeuralWorldEnv
from models import rollout
from models.state_mlp import load_checkpoint

model, codec, _ = load_checkpoint("artifacts/v0.2/state_mlp.pt")
env = NeuralWorldEnv()
observation, _ = env.reset(seed=42)
state = torch.from_numpy(codec.encode(observation))
states = rollout(model, codec, state, [Action.RIGHT, Action.DOWN] * 5)
env.close()
# states: 初期状態を含む11状態。実環境ではなくモデル自身の予測を次の入力に使う。
```

アプリ表示は開発版 `v0.3.0.dev1`。次は長期予測の精度評価・可視化、行動計画（v0.4）、適応型音楽（v0.5）へ進む予定です。
