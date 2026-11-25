import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export_onnx_and_profile(model: torch.nn.Module, sample_input: torch.Tensor, onnx_path: str, runs: int = 80) -> dict:
    """Exports trained model to ONNX and profiles sub-second edge inference latency."""
    model_cpu = model.cpu().eval()
    sample_cpu = sample_input[:1].cpu()
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    onnx_exported = False
    onnx_err = ""
    try:
        torch.onnx.export(
            model_cpu,
            sample_cpu,
            onnx_path,
            input_names=["traffic_window"],
            output_names=["multi_horizon_speed"],
            dynamic_axes={"traffic_window": {0: "batch"}, "multi_horizon_speed": {0: "batch"}},
            opset_version=14,
        )
        onnx_exported = os.path.exists(onnx_path)
    except Exception as e:
        onnx_err = str(e)

    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(sample_cpu)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model_cpu(sample_cpu)
        pt_ms = (time.perf_counter() - t0) / runs * 1000.0

    profile = {
        "onnx_exported": onnx_exported,
        "onnx_file": os.path.basename(onnx_path) if onnx_exported else None,
        "onnx_size_kb": round(os.path.getsize(onnx_path) / 1024.0, 2) if onnx_exported else 0.0,
        "pytorch_edge_latency_ms": round(pt_ms, 3),
        "sub_second_compliant": pt_ms < 1000.0,
    }
    if not onnx_exported:
        profile["onnx_export_note"] = onnx_err

    try:
        if onnx_exported:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            inp_np = sample_cpu.numpy()
            for _ in range(10):
                sess.run(None, {"traffic_window": inp_np})
            t0 = time.perf_counter()
            for _ in range(runs):
                sess.run(None, {"traffic_window": inp_np})
            profile["onnxruntime_edge_latency_ms"] = round((time.perf_counter() - t0) / runs * 1000.0, 3)
    except Exception as e:
        profile["onnxruntime_note"] = str(e)

    with open(onnx_path.replace(".onnx", "_latency.json"), "w") as f:
        json.dump(profile, f, indent=2)
    return profile


def generate_spatial_congestion_map(coords: np.ndarray, sensor_ids: list, pred_speeds: np.ndarray, out_html: str):
    """Generates interactive spatial congestion heatmap (Plotly if available, standalone Leaflet HTML fallback)."""
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    try:
        import plotly.express as px
        df_map = pd.DataFrame({
            "sensor_id": sensor_ids,
            "latitude": coords[:, 0],
            "longitude": coords[:, 1],
            "predicted_speed_mph": np.round(pred_speeds, 2),
            "congestion_level": np.where(pred_speeds < 35, "Severe Congestion",
                                np.where(pred_speeds < 52, "Moderate Traffic", "Free Flow")),
        })
        fig = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            color="predicted_speed_mph",
            size=np.clip(70.0 - df_map["predicted_speed_mph"], 8, 35),
            hover_name="sensor_id",
            hover_data=["congestion_level", "predicted_speed_mph"],
            color_continuous_scale="RdYlGn",
            range_color=[15, 70],
            zoom=9.5,
            mapbox_style="carto-positron",
            title="LA Urban Network 30-Min Predicted Speed & Congestion Heatmap (RA-DynSTGCN)",
        )
        fig.write_html(out_html)
    except Exception:
        markers = []
        for sid, (lat, lon), spd in zip(sensor_ids, coords, pred_speeds):
            color = "#d73027" if spd < 35 else ("#fc8d59" if spd < 52 else "#1a9850")
            markers.append(f'L.circleMarker([{lat:.5f},{lon:.5f}],{{radius:6,color:"{color}",fillOpacity:0.85}}).bindPopup("Sensor {sid}: {spd:.1f} mph").addTo(map);')
        html = (
            '<!DOCTYPE html><html><head><meta charset="utf-8"><title>LA Traffic Congestion Heatmap</title>'
            '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
            '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
            '<style>body{margin:0;font-family:sans-serif}#map{height:92vh}.hdr{padding:10px 16px;background:#111;color:#fff}</style></head>'
            '<body><div class="hdr"><b>LA Urban Network 30-Min Predicted Speed & Congestion Heatmap (RA-DynSTGCN)</b></div>'
            '<div id="map"></div><script>var map=L.map("map").setView([34.05,-118.25],10);'
            'L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png").addTo(map);'
            + ''.join(markers) + '</script></body></html>'
        )
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)


def plot_benchmark_figures(df_summary: pd.DataFrame, dyn_adj: np.ndarray, coords: np.ndarray,
                           pred_all: np.ndarray, true_all: np.ndarray, out_dir: str):
    """Generates publication-ready PNG diagnostic figures."""
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    datasets = df_summary["Dataset"].unique()
    x = np.arange(3)
    width = 0.15
    for ax, ds in zip(axes, datasets):
        sub = df_summary[df_summary["Dataset"] == ds]
        for idx, row in enumerate(sub.itertuples(index=False)):
            vals = [row._7, row._8, row._9]
            short_name = row.Model.split("_", 1)[1]
            ax.bar(x + (idx - 2) * width, vals, width, label=short_name)
        ax.set_title(ds.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["15-Min Horizon", "30-Min Horizon", "60-Min Horizon"])
        ax.set_ylabel("Mean Absolute Error (MAE, mph)")
        ax.legend(fontsize=8, loc="upper left")
    plt.suptitle("5-Model Multi-Horizon Traffic Forecasting Benchmark Across Dataset01 & Dataset02", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_dataset_horizon_mae.png"), dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(dyn_adj[:40, :40], cmap="viridis", aspect="auto")
    ax.set_title("Learned Regime-Aware Dynamic Adjacency (First 40 Sensors)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Target Sensor Index")
    ax.set_ylabel("Source Sensor Index")
    plt.colorbar(im, ax=ax, label="Normalized Edge Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dynamic_adjacency_heatmap.png"), dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if coords is not None:
        sc = axes[0].scatter(coords[:, 1], coords[:, 0], c=pred_all[0, 5, :], cmap="RdYlGn", vmin=20, vmax=68, s=36, edgecolors="k", linewidths=0.3)
        axes[0].set_title("LA Highway Sensor Network: 30-Min Forecast (mph)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")
        plt.colorbar(sc, ax=axes[0], label="Speed (mph)")

    steps = np.arange(1, 13) * 5
    axes[1].plot(steps, true_all[0, :, 10], "o-", label="Ground Truth (Sensor #10)", linewidth=2)
    axes[1].plot(steps, pred_all[0, :, 10], "s--", label="RA-DynSTGCN Forecast", linewidth=2)
    axes[1].set_title("Multi-Horizon Speed Trajectory (5 to 60 Min Ahead)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Forecast Horizon (Minutes)")
    axes[1].set_ylabel("Traffic Speed (mph)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spatial_and_trajectory_forecast.png"), dpi=180)
    plt.close()
