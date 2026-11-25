import torch
import torch.nn as nn


class BuiltInLSTMForecaster(nn.Module):
    """Built-In Model 1: Multi-Horizon Node-Shared LSTM (torch.nn.LSTM)."""
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, pred_len: int = 12, dropout: float = 0.15):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(in_channels, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        out, _ = self.lstm(x.permute(0, 2, 1, 3).reshape(B * N, T, C))
        return self.head(out[:, -1, :]).view(B, N, self.pred_len).permute(0, 2, 1)


class BuiltInGRUForecaster(nn.Module):
    """Built-In Model 2: Multi-Horizon GRU with Temporal Attention (torch.nn.GRU)."""
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, pred_len: int = 12, dropout: float = 0.15):
        super().__init__()
        self.pred_len = pred_len
        self.gru = nn.GRU(in_channels, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        out, _ = self.gru(x.permute(0, 2, 1, 3).reshape(B * N, T, C))
        weights = torch.softmax(self.attn(out), dim=1)
        ctx = torch.sum(out * weights, dim=1)
        pred = self.head(torch.cat([out[:, -1, :], ctx], dim=-1))
        return pred.view(B, N, self.pred_len).permute(0, 2, 1)


class BuiltInTransformerForecaster(nn.Module):
    """Built-In Model 3: Spatio-Temporal Transformer Encoder (torch.nn.TransformerEncoder)."""
    def __init__(self, num_nodes: int = 207, in_channels: int = 4, d_model: int = 24,
                 nhead: int = 4, pred_len: int = 12, dropout: float = 0.15):
        super().__init__()
        self.pred_len = pred_len
        self.input_proj = nn.Linear(in_channels, d_model)
        self.node_emb = nn.Parameter(torch.randn(1, 1, num_nodes, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, 12, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        h = self.input_proj(x) + self.node_emb[:, :, :N, :] + self.pos_emb[:, :T, :, :]
        enc = self.encoder(h.permute(0, 2, 1, 3).reshape(B * N, T, -1))
        return self.head(enc[:, -1, :]).view(B, N, self.pred_len).permute(0, 2, 1)
