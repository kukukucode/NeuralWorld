# NeuralWorld

ゲーム世界の変化をAIに学習させ、将来的に未来予測・行動計画・音楽表現へつなげる実験プロジェクトです。

**v0.4** では、学習したState World Modelの予測を使って、AIが2Dゲームの次の一手を選びます。

## できること

- プレイヤー・壁・箱・ゴールのある2Dゲームをプレイ
- 同じseedと操作列から同じ結果を再現
- ゲームデータを生成し、PyTorchのMLPで次状態を学習・評価
- 操作列に沿って予測を繰り返す（Goal・壁は固定、Player・Boxを更新）
- 1・5・10・20・50ステップ後の誤差を測定し、グラフを保存
- Random Shootingで操作列を比較し、ゴールに近づく一手を選択

## セットアップ

Python 3.10以上が必要です。

```bash
python -m pip install -e ".[dev]"
```

## 実行

```bash
python -m game.play --seed 42              # ゲームをプレイ（矢印キー / WASD）
python -m experiments.one_step --regenerate # データ生成・学習・1-step評価
python -m experiments.rollout             # 連続予測の評価・グラフ生成
python -m experiments.planning            # AIの行動計画とランダム操作を比較
pytest                                     # テスト
```

学習設定は `configs/state_mlp.toml`。学習結果は `artifacts/v0.2/`、連続予測の数値とグラフは `artifacts/v0.3/` に保存されます。初回は先に学習コマンドを実行してください。評価はテスト用エピソードの同じ開始地点を使い、最大512件（全件は `--max-windows 0`）で比較します。

行動計画では、観測した壁を避けたゴールまでの距離で予測を採点し、最初の一手を実行するたびに計画を立て直します。既存3マップで同じseedと手数上限を使い、ランダム操作と1・5・10手先の計画を比較します。到達率・成功時の平均手数・衝突率などは `artifacts/v0.4/metrics.json` に保存されます。

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

アプリ表示は `v0.4.0`。次は適応型音楽（v0.5）へ進む予定です。

## 研究計画：自己主導探索と経験記憶（MIT Ekoの着想）

### 目的と研究上の位置づけ

MITの [Is this machine playing?（Eko）](https://arxiv.org/abs/2610.07130) は、明示的な達成目標を与えず、シミュレーション世界でエージェントに探索させた研究です。体験をエピソード記憶と意味記憶に保持し、探索経験のないエージェントとの課題比較や、記憶の一部を取り除く比較で知識の再利用を調べています（[研究サイト](https://machineplaying.org/)）。

NeuralWorldでは、島や3Dエンジンをすぐ移植するのではなく、既存の**決定的な2D環境とState World Modelを保持したまま**、次の循環を研究します。

`観察 → 自分で探索対象を選ぶ → 行動 → 予測と結果を比較 → 仮説・技術を記憶 → 別の課題で再利用`

自己主導探索・永続記憶は今後の研究計画です。v0.4ではWorld Modelを使った行動計画の基準としてRandom Shootingを実装しました。Ekoの外部記憶による経験学習と、NeuralWorldのMLPの重み更新による状態予測学習は区別して実装・評価します。

### 段階的な組み込み方

1. **v0.3系：予測の検証を先に整える。** 既存の `models/rollout.py` で、1-stepだけでなく複数ステップ後の位置誤差、衝突・箱操作時の失敗、合法なグリッド状態からの逸脱を測定する。既存の `game/physics.py` を真の遷移として比較し、予測を実環境の正解と混同しない。
2. **探索エージェントの最小実装。** 現在の `NeuralWorldEnv` に対して、ランダム行動を基準とし、未訪問状態・操作の多様性・予測改善量などで行動を選ぶ探索モードを別モジュールとして追加する。単なる予測誤差最大化は、学習不能な状態への固着を招き得るため、反復制限と探索予算を設ける。初期段階でLLMや3D化は必須としない。
3. **永続記憶を追加。** 行動・観測・予測・結果・シード・モデル版をエピソード記憶（JSONL）に記録し、再現できた法則や操作手順を意味記憶（Markdown）として抽出する。意味記憶には `hypothesis / verified / contradicted` のような状態と根拠となるエピソードIDを付け、誤った法則を修正できるようにする。実行ログは生成物としてGit管理対象から分離する。
4. **記憶を使った行動計画への拡張。** v0.4のRandom Shootingを基準に、保存した技術を検索して行動を比較する。予測が不確実、状態が不正、計算予算を超える場合は安全な既存方策へ戻す。提案を記録するだけの `shadow` モードから始め、効果を確認後にのみ行動へ反映する。
5. **環境拡張は最後に検討。** 現在の移動・箱・壁・ゴールで学習と再利用を確認してから、持つ・置く・組み合わせる・スイッチを操作するなどの行動を少しずつ追加する。MITのClawblox/3D世界への移行は、2Dでは検証できない研究課題が明確になった場合だけ判断する。

想定する新規コンポーネントは探索方策、経験ロガー、知識の検証・更新、記憶検索、計画器、評価ランナーです。具体的なファイル配置は実装時に決め、現段階では既存の `game/`、`data/`、`models/`、`experiments/` とテストを壊さないことを優先します。

### 評価と採用基準

| 比較条件 | 何を切り分けるか |
|---|---|
| ランダム探索・記憶なし | 最低限の基準 |
| 好奇心探索・記憶なし | 探索方策による差 |
| 好奇心探索・記憶あり | 記憶の追加価値 |
| 好奇心探索・記憶あり・モデル計画あり | 予測による計画の追加価値 |

探索ステップ数、観測できる情報、使用可能な行動、CPU時間の上限をそろえ、探索に使用していないマップ・シード・課題で比較します。ゴール到達率、操作成功率、必要行動数、複数ステップ予測誤差、記憶内容の再現率、CPU時間・メモリ使用量を記録します。さらに記憶を削除・破損させる比較を行い、成功が保存した知識に依存するかを確認します。改善が確認できない構成は採用せず、失敗結果も残します。

参考となる既存研究：[Curiosity-driven Exploration by Self-supervised Prediction（ICML 2017）](https://proceedings.mlr.press/v70/pathak17a.html)、[Plan2Explore（ICML 2020）](https://proceedings.mlr.press/v119/sekar20a.html)、[Go-Explore（Nature 2021）](https://doi.org/10.1038/s41586-020-03157-9)。それぞれの結果は当該論文の環境で得られたもので、NeuralWorldでの改善を保証するものではありません。
