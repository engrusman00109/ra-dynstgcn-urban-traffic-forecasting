from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExperimentConfig:
    seed: int = 42
    num_nodes: int = 207
    seq_len: int = 12          # 12 * 5 min = 60 min historical window
    pred_len: int = 12         # 12 * 5 min = 60 min forecast window
    eval_horizons: Tuple[int, int, int] = (2, 5, 11)  # 15-min, 30-min, 60-min
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    batch_size: int = 32
    epochs: int = 6
    lr: float = 3e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    hidden_dim: int = 24
    embed_dim: int = 16
    diffusion_steps: int = 2
    dropout: float = 0.15
    train_stride: int = 12
    eval_stride: int = 12
    device: str = "cuda"
