# NeuralWorld

ゲーム世界の変化をAIに学習させ、将来的に未来予測・行動計画・音楽表現へつなげる実験プロジェクトです。

現在の **v0.2** では、2Dゲーム環境と、次の状態を予測する State World Model を実装しています。音楽生成は今後追加する予定です。

## できること

- プレイヤー・壁・箱・ゴールのある2Dゲームをプレイ
- 同じseedと操作列から同じ結果を再現
- ゲームデータを生成し、PyTorchのMLPで次状態を学習・評価

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

次は長期予測（v0.3）、行動計画（v0.4）、適応型音楽（v0.5）へ進む予定です。
