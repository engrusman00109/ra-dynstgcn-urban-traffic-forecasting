import os
import gc
import shutil
import zipfile
import pickle
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class SpatioTemporalGraphLoader:
    """
    Unified dataset acquisition, spatial graph fusion, missing-value imputation,
    and chronological window generator for Dataset01 and Dataset02.
    """

    def __init__(self, extract_root: str = "/tmp/stgnn_extract"):
        self.extract_root = extract_root
        os.makedirs(self.extract_root, exist_ok=True)

    @staticmethod
    def _impute_and_smooth(speed_raw: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        1. Treats exact 0.0 mph readings as sensor dropouts (NaN).
        2. Applies temporal linear interpolation per sensor.
        3. Applies graph-neighborhood weighted mean smoothing for residual gaps.
        """
        orig_zero_mask = (speed_raw == 0.0)
        df = pd.DataFrame(speed_raw)
        df.replace(0.0, np.nan, inplace=True)
        imputed = df.interpolate(method="linear", axis=0, limit_direction="both").values.astype(np.float32)
        del df

        row_sums = adj.sum(axis=1, keepdims=True)
        norm_adj = np.divide(adj, np.maximum(row_sums, 1e-6))
        nan_mask = np.isnan(imputed)
        if nan_mask.any():
            filled_temp = np.nan_to_num(imputed, nan=float(np.nanmean(imputed)))
            neighbor_mean = filled_temp @ norm_adj.T
            imputed[nan_mask] = neighbor_mean[nan_mask]

        neighbor_all = imputed @ norm_adj.T
        imputed[orig_zero_mask] = 0.90 * imputed[orig_zero_mask] + 0.10 * neighbor_all[orig_zero_mask]
        return imputed.astype(np.float32)

    @staticmethod
    def _compute_time_features(timestamps: np.ndarray) -> np.ndarray:
        dt = pd.to_datetime(timestamps)
        tod = (dt.hour * 60 + dt.minute) / (24.0 * 60.0)
        dow = dt.dayofweek / 7.0
        return np.stack([
            np.sin(2 * np.pi * tod).astype(np.float32),
            np.cos(2 * np.pi * tod).astype(np.float32),
            np.sin(2 * np.pi * dow).astype(np.float32),
        ], axis=-1)

    def load_dataset_01(self, zip_path: str) -> Dict:
        target_dir = os.path.join(self.extract_root, "dataset01")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        with h5py.File(os.path.join(target_dir, "METR-LA.h5"), "r") as f:
            raw_speed = f["df"]["block0_values"][:].astype(np.float32)
            timestamps = f["df"]["axis1"][:]
            sensor_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["df"]["axis0"][:]]

        with open(os.path.join(target_dir, "adj_METR-LA.pkl"), "rb") as f:
            _, _, adj_mx = pickle.load(f, encoding="latin1")
        adj_mx = adj_mx.astype(np.float32)
        np.fill_diagonal(adj_mx, 1.0)

        clean_speed = self._impute_and_smooth(raw_speed, adj_mx)
        del raw_speed
        gc.collect()
        shutil.rmtree(target_dir, ignore_errors=True)

        return {
            "name": "Dataset01_Physical_RoadGraph",
            "speed": clean_speed,
            "adj": adj_mx,
            "time_feats": self._compute_time_features(timestamps),
            "sensor_ids": sensor_ids,
            "spatial_emb": None,
            "coords": None,
        }

    def load_dataset_02(self, zip_path: str) -> Dict:
        target_dir = os.path.join(self.extract_root, "dataset02")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

        with h5py.File(os.path.join(target_dir, "metr-la.h5"), "r") as f:
            raw_speed = f["df"]["block0_values"][:].astype(np.float32)
            timestamps = f["df"]["axis1"][:]
            sensor_ids = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in f["df"]["axis0"][:]]

        with open(os.path.join(target_dir, "adj_mx.pkl"), "rb") as f:
            _, id_to_idx, adj_mx = pickle.load(f, encoding="latin1")
        adj_mx = adj_mx.astype(np.float32)

        # Parse Node2Vec Spatial Embeddings (SE_metrla.txt)
        with open(os.path.join(target_dir, "SE_metrla.txt"), "r") as f:
            header = f.readline().strip().split()
            se_dim = int(header[1])
            se_matrix = np.zeros((len(sensor_ids), se_dim), dtype=np.float32)
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) < len(sensor_ids):
                    se_matrix[int(parts[0])] = np.array([float(v) for v in parts[1:]], dtype=np.float32)

        se_norm = se_matrix / np.maximum(np.linalg.norm(se_matrix, axis=1, keepdims=True), 1e-6)
        se_sim = np.clip(se_norm @ se_norm.T, 0.0, 1.0)

        # Geodesic distance Gaussian kernel from distances_la_2012.csv
        df_dist = pd.read_csv(os.path.join(target_dir, "distances_la_2012.csv"))
        dist_mat = np.full((len(sensor_ids), len(sensor_ids)), np.inf, dtype=np.float32)
        np.fill_diagonal(dist_mat, 0.0)
        for row in df_dist.itertuples(index=False):
            u, v, c = str(row[0]), str(row[1]), float(row[2])
            if u in id_to_idx and v in id_to_idx:
                dist_mat[id_to_idx[u], id_to_idx[v]] = c
        del df_dist
        finite_dists = dist_mat[np.isfinite(dist_mat) & (dist_mat > 0)]
        sigma = float(np.std(finite_dists)) if len(finite_dists) > 0 else 1000.0
        geo_adj = np.exp(-np.square(dist_mat / max(sigma, 1.0)))
        geo_adj[geo_adj < 0.1] = 0.0

        # Fused Multi-Modal Adjacency Matrix
        hybrid_adj = 0.55 * adj_mx + 0.25 * geo_adj + 0.20 * (se_sim * (adj_mx > 0).astype(np.float32))
        np.fill_diagonal(hybrid_adj, 1.0)

        df_loc = pd.read_csv(os.path.join(target_dir, "graph_sensor_locations.csv")).sort_values("index")
        coords = df_loc[["latitude", "longitude"]].values.astype(np.float32)

        clean_speed = self._impute_and_smooth(raw_speed, hybrid_adj)
        del raw_speed
        gc.collect()
        shutil.rmtree(target_dir, ignore_errors=True)

        return {
            "name": "Dataset02_MultiModal_SpatialTopology",
            "speed": clean_speed,
            "adj": hybrid_adj.astype(np.float32),
            "time_feats": self._compute_time_features(timestamps),
            "sensor_ids": sensor_ids,
            "spatial_emb": se_matrix,
            "coords": coords,
        }


class TrafficWindowDataset(Dataset):
    def __init__(self, speed: np.ndarray, time_feats: np.ndarray, mean: float, std: float,
                 seq_len: int = 12, pred_len: int = 12, stride: int = 12):
        self.speed_norm = ((speed - mean) / std).astype(np.float32)
        self.time_feats = time_feats.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        max_start = len(speed) - seq_len - pred_len + 1
        self.indices = np.arange(0, max_start, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.indices[idx]
        x_speed = self.speed_norm[s : s + self.seq_len]
        t_feat = self.time_feats[s : s + self.seq_len]
        N = x_speed.shape[1]
        t_broad = np.broadcast_to(t_feat[:, None, :], (self.seq_len, N, 3))
        x = np.concatenate([x_speed[..., None], t_broad], axis=-1)
        y = self.speed_norm[s + self.seq_len : s + self.seq_len + self.pred_len]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def create_dataloaders(data_dict: Dict, seq_len: int = 12, pred_len: int = 12, batch_size: int = 32,
                       train_ratio: float = 0.70, val_ratio: float = 0.10,
                       train_stride: int = 12, eval_stride: int = 12):
    speed, time_feats = data_dict["speed"], data_dict["time_feats"]
    T = len(speed)
    n_train = int(T * train_ratio)
    n_val = int(T * (train_ratio + val_ratio))

    mean, std = float(speed[:n_train].mean()), float(speed[:n_train].std())
    train_ds = TrafficWindowDataset(speed[:n_train], time_feats[:n_train], mean, std, seq_len, pred_len, train_stride)
    val_ds = TrafficWindowDataset(speed[n_train:n_val], time_feats[n_train:n_val], mean, std, seq_len, pred_len, eval_stride)
    test_ds = TrafficWindowDataset(speed[n_val:], time_feats[n_val:], mean, std, seq_len, pred_len, eval_stride)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        {"mean": mean, "std": std},
    )
