import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DynamicHybridGraphConstructor(nn.Module):
    """
    Research Core Module 1: Dynamic Hybrid Graph Construction.
    Fuses physical forward/backward random-walk matrices (P_f, P_b),
    learnable adaptive node embeddings (E1, E2), and a traffic-regime-conditioned
    dynamic edge attention gate.
    """
    def __init__(self, adj_mx: np.ndarray, num_nodes: int = 207, embed_dim: int = 16,
                 hidden_dim: int = 24, pretrained_se: Optional[np.ndarray] = None):
        super().__init__()
        adj = adj_mx.astype(np.float32)
        p_fwd = adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-6)
        p_bwd = adj.T / np.maximum(adj.T.sum(axis=1, keepdims=True), 1e-6)
        self.register_buffer("p_fwd", torch.from_numpy(p_fwd))
        self.register_buffer("p_bwd", torch.from_numpy(p_bwd))

        if pretrained_se is not None:
            se_tensor = torch.from_numpy(pretrained_se[:, :embed_dim].astype(np.float32))
            self.node_vec1 = nn.Parameter(se_tensor.clone())
            self.node_vec2 = nn.Parameter(se_tensor.clone().T)
        else:
            self.node_vec1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
            self.node_vec2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.1)

        self.regime_proj = nn.Linear(hidden_dim, embed_dim)
        self.alpha_logits = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))

    def forward(self, node_context: torch.Tensor):
        a_adapt = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=-1)
        q = self.regime_proj(node_context)
        scores = torch.matmul(q, q.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        support_mask = ((self.p_fwd + self.p_bwd) > 0).float().unsqueeze(0)
        a_state = F.softmax(scores.masked_fill(support_mask == 0, -1e9), dim=-1)

        w = F.softmax(self.alpha_logits, dim=0)
        a_dyn = w[0] * a_adapt.unsqueeze(0) + w[1] * a_state + w[2] * self.p_fwd.unsqueeze(0)
        return self.p_fwd, self.p_bwd, a_dyn


class MultiScaleDiffusionGraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, diffusion_steps: int = 2, dropout: float = 0.15):
        super().__init__()
        self.K = diffusion_steps
        num_supports = 1 + 2 * diffusion_steps + 1
        self.mlp = nn.Conv2d(c_in * num_supports, c_out, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, p_fwd: torch.Tensor, p_bwd: torch.Tensor, a_dyn: torch.Tensor) -> torch.Tensor:
        supports = [x]
        x_f, x_b = x, x
        for _ in range(self.K):
            x_f = torch.einsum("bcnt,nm->bcmt", x_f, p_fwd)
            x_b = torch.einsum("bcnt,nm->bcmt", x_b, p_bwd)
            supports.extend([x_f, x_b])
        supports.append(torch.einsum("bcnt,bnm->bcmt", x, a_dyn))
        return self.dropout(self.mlp(torch.cat(supports, dim=1)))


class GatedDilatedTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation))
        self.gate_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), dilation=(1, dilation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.padding, 0))
        return self.dropout(torch.tanh(self.filter_conv(x_pad)) * torch.sigmoid(self.gate_conv(x_pad)))


class CustomDynamicSTGCN(nn.Module):
    """Proposed Research Model (Built from Scratch): RA-DynSTGCN."""
    def __init__(self, adj_mx: np.ndarray, num_nodes: int = 207, in_channels: int = 4,
                 hidden_dim: int = 24, embed_dim: int = 16, pred_len: int = 12,
                 diffusion_steps: int = 2, dropout: float = 0.15, pretrained_se: Optional[np.ndarray] = None):
        super().__init__()
        self.start_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1))
        self.graph_builder = DynamicHybridGraphConstructor(adj_mx, num_nodes, embed_dim, hidden_dim, pretrained_se)

        self.tcn1 = GatedDilatedTCNBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout)
        self.gcn1 = MultiScaleDiffusionGraphConv(hidden_dim, hidden_dim, diffusion_steps, dropout)
        self.norm1 = nn.BatchNorm2d(hidden_dim)

        self.tcn2 = GatedDilatedTCNBlock(hidden_dim, kernel_size=3, dilation=2, dropout=dropout)
        self.gcn2 = MultiScaleDiffusionGraphConv(hidden_dim, hidden_dim, diffusion_steps, dropout)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

        self.skip_conv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(hidden_dim * 2, pred_len, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.start_conv(x.permute(0, 3, 2, 1))
        p_fwd, p_bwd, a_dyn = self.graph_builder(h0.mean(dim=-1).permute(0, 2, 1))

        h1 = self.norm1(h0 + self.gcn1(self.tcn1(h0), p_fwd, p_bwd, a_dyn))
        h2 = self.norm2(h1 + self.gcn2(self.tcn2(h1), p_fwd, p_bwd, a_dyn))

        skip = torch.cat([h1[..., -1:], h2[..., -1:]], dim=1)
        out = F.gelu(self.skip_conv(skip))
        out = F.gelu(self.end_conv_1(out))
        return self.end_conv_2(out).squeeze(-1)

    @torch.no_grad()
    def extract_dynamic_adjacency(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        h0 = self.start_conv(x.permute(0, 3, 2, 1))
        _, _, a_dyn = self.graph_builder(h0.mean(dim=-1).permute(0, 2, 1))
        return a_dyn.mean(dim=0).cpu().numpy()
