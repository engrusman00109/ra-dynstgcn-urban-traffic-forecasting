import torch
import torch.nn as nn
import torch.nn.functional as F
from models.builtin_models import (
    BuiltInLSTMForecaster,
    BuiltInGRUForecaster,
    BuiltInTransformerForecaster,
)


class HorizonAdaptiveEnsembleForecaster(nn.Module):
    """
    Model 5: Combines the 3 built-in models (LSTM + GRU-Attention + Transformer)
    via a learned horizon- and traffic-regime-aware softmax gating network.
    """
    def __init__(self, num_nodes: int = 207, in_channels: int = 4, hidden_dim: int = 32,
                 pred_len: int = 12, dropout: float = 0.15):
        super().__init__()
        self.branch_lstm = BuiltInLSTMForecaster(in_channels, hidden_dim, pred_len, dropout)
        self.branch_gru = BuiltInGRUForecaster(in_channels, hidden_dim, pred_len, dropout)
        self.branch_trans = BuiltInTransformerForecaster(num_nodes, in_channels, 24, 4, pred_len, dropout)

        self.horizon_gate = nn.Parameter(torch.zeros(1, pred_len, num_nodes, 3))
        self.ctx_gate = nn.Sequential(nn.Linear(in_channels, 16), nn.GELU(), nn.Linear(16, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([self.branch_lstm(x), self.branch_gru(x), self.branch_trans(x)], dim=-1)
        ctx_logits = self.ctx_gate(x[:, -1, :, :]).unsqueeze(1)
        weights = F.softmax(self.horizon_gate + ctx_logits, dim=-1)
        return torch.sum(stacked * weights, dim=-1)
