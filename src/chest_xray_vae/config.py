from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    root: str
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    split: str = "test"
    max_eval_images: Optional[int] = None


@dataclass
class ModelConfig:
    in_channels: int = 1
    latent_dim: int = 128
    hidden_dims: tuple[int, ...] = (32, 64, 128, 256)


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 2e-4
    beta: float = 0.25
    beta_warmup_epochs: int = 20
    device: str = "auto"


@dataclass
class OutputConfig:
    exp_dir: str = "outputs/train_run"
    save_every: int = 5


@dataclass
class EvalConfig:
    checkpoint_path: str = "outputs/train_run/checkpoints/best.pt"
    out_dir: str = "outputs/eval_run"
    num_fake_mult: int = 1
    device: str = "auto"


@dataclass
class FullTrainConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output: OutputConfig


@dataclass
class FullEvalConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    eval: EvalConfig


def _normalize_hidden_dims(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    raise TypeError(f"hidden_dims must be a list or tuple, got {type(value)!r}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_config(path: str | Path) -> FullTrainConfig:
    raw = load_yaml(path)
    return FullTrainConfig(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(
            **{**raw["model"], "hidden_dims": _normalize_hidden_dims(raw["model"]["hidden_dims"])}
        ),
        train=TrainConfig(**raw["train"]),
        output=OutputConfig(**raw["output"]),
    )


def load_eval_config(path: str | Path) -> FullEvalConfig:
    raw = load_yaml(path)
    return FullEvalConfig(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(
            **{**raw["model"], "hidden_dims": _normalize_hidden_dims(raw["model"]["hidden_dims"])}
        ),
        eval=EvalConfig(**raw["eval"]),
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)
