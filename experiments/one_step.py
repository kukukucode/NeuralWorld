"""Train and evaluate the v0.2 one-step State MLP baseline."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import torch

from data.dataset import load_dataset
from data.generate import generate_dataset
from models.state_mlp import (
    StateMLP,
    TrainingConfig,
    evaluate_model,
    save_checkpoint,
    train_model,
)


def run_experiment(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    regenerate: bool = False,
) -> dict[str, Any]:
    """Generate data, train the MLP, and evaluate only on the held-out test split."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    dataset_config = config["dataset"]
    model_config = config["model"]
    training_config = TrainingConfig(**config["training"])
    dataset_path = output_dir / "transitions.npz"
    if regenerate or not dataset_path.exists():
        generate_dataset(dataset_path, **dataset_config)

    bundle = load_dataset(dataset_path)
    torch.manual_seed(training_config.seed)
    model = StateMLP(
        bundle.codec.state_size,
        target_size=bundle.codec.target_size,
        hidden_sizes=tuple(model_config["hidden_sizes"]),
    )
    training_result = train_model(
        model,
        bundle.train,
        bundle.validation,
        training_config,
    )
    test_metrics = evaluate_model(model, bundle.test, bundle.codec)
    checkpoint_path = save_checkpoint(
        output_dir / "state_mlp.pt",
        model,
        codec=bundle.codec,
        training_config=training_config,
    )
    result: dict[str, Any] = {
        "version": "0.2.0",
        "dataset": bundle.metadata,
        "splits": {
            "train": len(bundle.train),
            "validation": len(bundle.validation),
            "test": len(bundle.test),
        },
        "training": {
            "best_epoch": training_result.best_epoch + 1,
            "best_validation_mse": min(training_result.validation_loss),
            "final_train_mse": training_result.train_loss[-1],
        },
        "test": test_metrics,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeuralWorld v0.2 baseline")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/state_mlp.toml")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/v0.2"))
    parser.add_argument(
        "--regenerate", action="store_true", help="replace an existing dataset"
    )
    args = parser.parse_args()
    result = run_experiment(
        config_path=args.config,
        output_dir=args.output,
        regenerate=args.regenerate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
