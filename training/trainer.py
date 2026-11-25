import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    diff = preds - targets
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mask = targets > 5.0
    mape = float(np.mean(np.abs(diff[mask] / targets[mask])) * 100.0) if np.any(mask) else 0.0
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


class MultiHorizonTrainer:
    def __init__(self, model: nn.Module, device: torch.device, lr: float = 3e-3,
                 weight_decay: float = 1e-4, epochs: int = 6, huber_delta: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.criterion = nn.HuberLoss(delta=huber_delta)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1), eta_min=1e-5)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, scaler: Dict = None) -> Dict:
        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        history = {"train_loss": [], "val_loss": []}
        t0 = time.time()

        for _ in range(1, self.epochs + 1):
            self.model.train()
            tr_losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                tr_losses.append(loss.item())
            self.scheduler.step()

            self.model.eval()
            va_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    va_losses.append(self.criterion(self.model(x), y).item())

            tr_l, va_l = float(np.mean(tr_losses)), float(np.mean(va_losses))
            history["train_loss"].append(tr_l)
            history["val_loss"].append(va_l)
            if va_l < best_val_loss:
                best_val_loss = va_l
                best_weights = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_weights)
        history["train_time_sec"] = round(time.time() - t0, 2)
        return history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, scaler: Dict,
                 eval_horizons: Tuple[int, int, int] = (2, 5, 11)):
        self.model.eval()
        preds_list, targets_list = [], []
        mean, std = scaler["mean"], scaler["std"]
        for x, y in test_loader:
            out = self.model(x.to(self.device)).cpu().numpy()
            preds_list.append(out * std + mean)
            targets_list.append(y.numpy() * std + mean)

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        results = {"Overall": compute_metrics(preds, targets)}
        for h_idx, label in zip(eval_horizons, ["15-min", "30-min", "60-min"]):
            results[label] = compute_metrics(preds[:, h_idx, :], targets[:, h_idx, :])
        return results, preds, targets
