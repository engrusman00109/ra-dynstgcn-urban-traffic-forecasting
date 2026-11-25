import os
import gc
import sys
import random
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from configs.config import ExperimentConfig
from data.loader import SpatioTemporalGraphLoader, create_dataloaders
from models.builtin_models import (
    BuiltInLSTMForecaster,
    BuiltInGRUForecaster,
    BuiltInTransformerForecaster,
)
from models.custom_dyn_stgcn import CustomDynamicSTGCN
from models.ensemble_model import HorizonAdaptiveEnsembleForecaster
from training.trainer import MultiHorizonTrainer
from evaluation.benchmark_and_export import (
    export_onnx_and_profile,
    generate_spatial_congestion_map,
    plot_benchmark_figures,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_five_models(cfg: ExperimentConfig, adj_mx: np.ndarray, spatial_emb=None):
    return {
        "1_BuiltIn_LSTM": BuiltInLSTMForecaster(4, 32, cfg.pred_len, cfg.dropout),
        "2_BuiltIn_GRU_Attn": BuiltInGRUForecaster(4, 32, cfg.pred_len, cfg.dropout),
        "3_BuiltIn_Transformer": BuiltInTransformerForecaster(cfg.num_nodes, 4, 24, 4, cfg.pred_len, cfg.dropout),
        "4_Proposed_RA_DynSTGCN": CustomDynamicSTGCN(
            adj_mx=adj_mx,
            num_nodes=cfg.num_nodes,
            in_channels=4,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            pred_len=cfg.pred_len,
            diffusion_steps=cfg.diffusion_steps,
            dropout=cfg.dropout,
            pretrained_se=spatial_emb,
        ),
        "5_HorizonAdaptive_Ensemble": HorizonAdaptiveEnsembleForecaster(
            cfg.num_nodes, 4, 32, cfg.pred_len, cfg.dropout
        ),
    }


def main():
    cfg = ExperimentConfig()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_tables = os.path.join(BASE_DIR, "outputs", "tables")
    out_ckpts = os.path.join(BASE_DIR, "outputs", "checkpoints")
    out_figs = os.path.join(BASE_DIR, "outputs", "figures")
    out_onnx = os.path.join(BASE_DIR, "outputs", "onnx")
    for d in [out_tables, out_ckpts, out_figs, out_onnx]:
        os.makedirs(d, exist_ok=True)

    d1_zip = os.path.join(BASE_DIR, "Dataset01.zip") if os.path.exists(os.path.join(BASE_DIR, "Dataset01.zip")) else "Dataset01.zip"
    d2_zip = os.path.join(BASE_DIR, "Dataset02.zip") if os.path.exists(os.path.join(BASE_DIR, "Dataset02.zip")) else "Dataset02.zip"

    loader = SpatioTemporalGraphLoader(extract_root="/tmp/stgnn_extract")
    dataset_specs = [
        ("Dataset01", loader.load_dataset_01, d1_zip),
        ("Dataset02", loader.load_dataset_02, d2_zip),
    ]

    all_rows = []
    d2_dyn_adj = None
    d2_coords = None
    d2_preds = None
    d2_targets = None

    for tag, load_fn, zip_file in dataset_specs:
        ds = load_fn(zip_file)
        ds_name = ds["name"]
        print(f"=== Running 5-Model Benchmark on {ds_name} ===")
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            ds, cfg.seq_len, cfg.pred_len, cfg.batch_size,
            cfg.train_ratio, cfg.val_ratio, cfg.train_stride, cfg.eval_stride
        )
        models = build_five_models(cfg, ds["adj"], ds["spatial_emb"])

        for model_name, model in models.items():
            trainer = MultiHorizonTrainer(model, device, cfg.lr, cfg.weight_decay, cfg.epochs, cfg.huber_delta)
            hist = trainer.fit(train_loader, val_loader, scaler)
            metrics, preds, targets = trainer.evaluate(test_loader, scaler, cfg.eval_horizons)
            torch.save(model.state_dict(), os.path.join(out_ckpts, f"{ds_name}_{model_name}.pt"))

            row = {
                "Dataset": ds_name,
                "Model": model_name,
                "Params": sum(p.numel() for p in model.parameters()),
                "Train_Time_s": hist["train_time_sec"],
                "Overall_MAE": metrics["Overall"]["MAE"],
                "Overall_RMSE": metrics["Overall"]["RMSE"],
                "Overall_MAPE": metrics["Overall"]["MAPE"],
                "15min_MAE": metrics["15-min"]["MAE"],
                "30min_MAE": metrics["30-min"]["MAE"],
                "60min_MAE": metrics["60-min"]["MAE"],
            }
            all_rows.append(row)
            print(f"{model_name:28s} | Overall MAE: {row['Overall_MAE']:.4f} | 15m: {row['15min_MAE']:.4f} | 30m: {row['30min_MAE']:.4f} | 60m: {row['60min_MAE']:.4f}")

            if model_name == "4_Proposed_RA_DynSTGCN" and tag == "Dataset02":
                sample_x, _ = next(iter(test_loader))
                d2_dyn_adj = model.extract_dynamic_adjacency(sample_x.to(device))
                d2_coords = ds["coords"]
                d2_preds = preds[:2].copy()
                d2_targets = targets[:2].copy()
                prof = export_onnx_and_profile(model, sample_x, os.path.join(out_onnx, "ra_dynstgcn_d2.onnx"))
                print("ONNX & Edge Latency Profile:", prof)
                if ds["coords"] is not None:
                    generate_spatial_congestion_map(
                        ds["coords"],
                        ds["sensor_ids"],
                        preds[0, 5, :],
                        os.path.join(out_figs, "la_30min_congestion_map.html"),
                    )

            del trainer, model, preds, targets
            gc.collect()

        del ds, train_loader, val_loader, test_loader, models
        gc.collect()

    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv(os.path.join(out_tables, "cross_dataset_5model_comparison.csv"), index=False)

    piv = df_summary.pivot(index="Model", columns="Dataset", values=["Overall_MAE", "Overall_RMSE", "Overall_MAPE", "15min_MAE", "30min_MAE", "60min_MAE"])
    piv.to_csv(os.path.join(out_tables, "dataset01_vs_dataset02_pivot.csv"))

    if d2_dyn_adj is not None:
        plot_benchmark_figures(df_summary, d2_dyn_adj, d2_coords, d2_preds, d2_targets, out_figs)

    print("Completed end-to-end pipeline successfully!")


if __name__ == "__main__":
    main()
