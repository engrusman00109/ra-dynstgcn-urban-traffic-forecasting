# Spatio-Temporal Graph Neural Networks for Urban Traffic Flow and Intelligent Driving Assistance

## Abstract

Rapid urbanization, expanding vehicle fleets, and complex roadway topologies have made real-time urban traffic forecasting and intelligent driving assistance systems (ADAS) critical pillars of modern smart cities. Traditional time-series models such as ARIMA and Vector Autoregression (VAR) struggle to capture non-linear congestion propagation across non-Euclidean road networks, while standard Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) are restricted to regular Euclidean grids or isolated sequences. This project presents a **Spatio-Temporal Graph Neural Network (ST-GNN)** framework engineered to jointly model spatial road-network dependencies and temporal traffic dynamics for multi-horizon traffic speed/flow forecasting and downstream intelligent driving assistance. By representing road sensors and intersections as graph nodes connected by directed spatial proximity and flow edges, the framework learns cascading congestion patterns over short- and medium-term horizons and translates network-level forecasts into actionable guidance cues (such as dynamic speed advisories, congestion-aware route/lane recommendations, and proactive headway/braking alerts) for connected and intelligent vehicles. Experimental evaluation across two widely recognized real-world traffic benchmarks—**METR-LA** and **PEMS-BAY**—demonstrates strong predictive accuracy, spatial awareness during peak-hour bottlenecks, and practical utility for intelligent driving support.

## Table of Contents

 1. [Project Title & Overview](#project-title--overview)

 2. [Problem Statement](#problem-statement)

 3. [System Architecture & Visual Diagram](#system-architecture--visual-diagram)

 4. [Data Acquisition](#data-acquisition)

    * [Dataset 1 (METR-LA)](#dataset-1-metr-la)

    * [Dataset 2 (PEMS-BAY)](#dataset-2-pems-bay)

 5. [Methodology](#methodology)

    * [Graph Formulation & Adjacency Matrix](#graph-formulation--adjacency-matrix)

    * [Spatial Graph Convolution Module](#spatial-graph-convolution-module)

    * [Temporal Sequence Modeling Module](#temporal-sequence-modeling-module)

    * [Intelligent Driving Assistance (ADAS) Decision Layer](#intelligent-driving-assistance-adas-decision-layer)

    * [Training Objective & Optimization](#training-objective--optimization)

 6. [Experimental Setup & Evaluation Metrics](#experimental-setup--evaluation-metrics)

 7. [Discussion](#discussion)

 8. [Summary](#summary)

 9. [Conclusions & Future Work](#conclusions--future-work)

10. [Repository Structure & Usage](#repository-structure--usage)

11. [Author](#author)

## Project Title & Overview

**Spatio-Temporal Graph Neural Networks for Urban Traffic Flow and Intelligent Driving Assistance** bridges macroscopic urban mobility analytics with vehicle-level driving support. In densely connected metropolitan corridors, a slowdown at a single highway merge or arterial junction can ripple upstream and downstream within minutes. Vehicles equipped with intelligent driving assistance need more than just immediate local sensor perception (such as radar, LiDAR, or onboard cameras); they benefit from anticipatory awareness of approaching bottlenecks, shockwaves, and multi-step speed profiles along their corridor.

This repository implements an end-to-end pipeline covering:

* **Graph Construction & Preprocessing:** Converting raw traffic sensor time series and geospatial sensor coordinates into spatial adjacency graphs and normalized multi-step sliding windows.

* **ST-GNN Forecasting Engine:** Combining spatial message-passing/diffusion graph operations with temporal sequence modeling (such as gated recurrent or temporal convolutional blocks) to predict future traffic speeds/flows across all network nodes simultaneously.

* **Intelligent Driving Assistance Module:** Mapping forecasted network states, speed gradients, and congestion probabilities into vehicle-facing assistance signals, eco-speed recommendations, and corridor risk indicators.

## Problem Statement

Urban traffic systems are inherently **spatio-temporal** and **graph-structured**:

1. **Non-Euclidean Spatial Dependencies:** Road networks are irregular graphs where connectivity depends on highway topology, road distance, directional flow, and intersection bottlenecks rather than straight-line Euclidean distance alone.

2. **Dynamic Temporal Evolution:** Traffic states exhibit strong diurnal periodicity (morning and evening rush hours), weekly patterns (weekday commuter surges vs. weekend leisure travel), and sudden non-periodic perturbations (accidents, lane closures, or weather events).

3. **Cascading Bottlenecks & Shockwaves:** Congestion rarely stays isolated at one sensor; queue spillbacks propagate backward upstream while downstream capacity constraints alter arrival rates.

4. **Gap Between Traffic Forecasting and Driving Assistance:** Many traffic forecasting models stop at raw error metrics (MAE, RMSE, MAPE) without showing how multi-horizon predictions can support intelligent driving assistance—such as smoother speed adaptation, avoiding sudden stop-and-go waves, and adjusting following distances or route/corridor choices ahead of congestion fronts.

Given historical traffic observations $\mathbf{X}^{(t - T_p + 1 : t)} \in \mathbb{R}^{N \times T_p \times F}$ across $N$ road sensors over $T_p$ past time steps with $F$ input features on a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$, the core forecasting problem is to learn a mapping function $f_\theta(\cdot)$ that predicts future traffic states $\hat{\mathbf{Y}}^{(t + 1 : t + T_f)} \in \mathbb{R}^{N \times T_f}$ over the next $T_f$ horizons, and subsequently derives driving assistance advisories $\mathcal{A}_i^{(t+1:t+T_f)}$ for ego vehicles along route/corridor nodes $i \in \mathcal{V}_{\text{route}}$.

## System Architecture & Visual Diagram

The system is structured into four modular stages: **Data Acquisition & Graph Builder**, **ST-GNN Spatio-Temporal Encoder-Decoder**, **Multi-Horizon Traffic Predictor**, and **Intelligent Driving Assistance (ADAS) Interface**.

```
+---------------------------------------------------------------------------------------------------+
|                        SPATIO-TEMPORAL TRAFFIC & ADAS PIPELINE ARCHITECTURE                       |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [METR-LA & PEMS-BAY]      [Sensor Coordinates & Road Distances]                                   |
|   (5-Min Speed/Flow)  --->   (Pairwise Road Network Distances)                                   |
|          |                                 |                                                      |
|          v                                 v                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 1. DATA PREPROCESSING & GRAPH CONSTRUCTION                                                  |  |
|  |    - Missing value imputation & Z-score normalization                                       |  |
|  |    - Time-of-day / day-of-week feature encoding                                             |  |
|  |    - Gaussian kernel thresholded Adjacency Matrix A (N x N)                                 |  |
|  |    - Sliding window tensors: X in R^(N x Tp x F)  -->  Y in R^(N x Tf)                     |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 2. ST-GNN CORE BLOCKS (Stacked L Layers)                                                    |  |
|  |    +---------------------------------------+   +----------------------------------------+   |  |
|  |    | Spatial Graph Module                  |   | Temporal Sequence Module               |   |  |
|  |    | - Message passing / Diffusion conv    |-->| - Gated / Recurrent / Temporal conv    |   |  |
|  |    | - Captures upstream/downstream flow   |   | - Captures diurnal & shockwave trends  |   |  |
|  |    +---------------------------------------+   +----------------------------------------+   |  |
|  |                         \                                   /                                 |  |
|  |                          +-------> Residual & LayerNorm <--+                                  |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 3. MULTI-HORIZON FORECASTING HEAD                                                           |  |
|  |    - Predicts future speeds/flows across all N sensors for horizons t+1 ... t+Tf             |  |
|  |    - Computes spatial speed drops (dV/dx) & temporal deceleration (dV/dt)                   |  |
|  +---------------------------------------------------------------------------------------------+  |
|          |                                                                                        |
|          v                                                                                        |
|  +---------------------------------------------------------------------------------------------+  |
|  | 4. INTELLIGENT DRIVING ASSISTANCE (ADAS) MODULE                                             |  |
|  |    - Bottleneck / Shockwave Ahead Warning (lookahead risk score along planned path)          |  |
|  |    - Smooth Speed Advisory / Eco-Cruise Profile (prevents abrupt braking in queues)            |  |
|  |    - Adaptive Following / Corridor & Lane Guidance Cues                                     |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+

```

## Data Acquisition

This project uses two benchmark traffic datasets collected from highway sensor networks at **5-minute intervals**, widely used in spatio-temporal graph neural network research.

### Dataset 1 (METR-LA)

* **Dataset Name:** `METR-LA` (Los Angeles County Highway Traffic Speed Dataset)

* **Source / Reference Links:**

  * Benchmark Repository & Data Utilities: <https://github.com/liyaguang/DCRNN>

  * GNN Traffic Benchmark Notebook & Resources: <https://github.com/RiccardoSpolaor/Verbal-Explanations-of-Spatio-Temporal-Graph-Neural-Networks-for-Traffic-Forecasting>

  * Kaggle Mirror / Reference: <https://www.kaggle.com/datasets/xiaohualu/metr-la-complete>

#### Dataset Description (METR-LA)

`METR-LA` records traffic speed readings from **207 loop detectors** placed across major highways and corridors in Los Angeles County (covering areas such as central Los Angeles, Glendale, Burbank, and extending west toward Sepulveda). Observations are sampled every **5 minutes** (288 intervals per day) across a multi-month period in 2012 (March to June 2012, totaling roughly 34,272 time steps). Alongside sensor speed values, supporting metadata includes sensor IDs, geographic coordinates (latitude and longitude), pairwise road distances, and a precomputed spatial adjacency matrix. A notable characteristic of `METR-LA` is the presence of missing or zero readings (\~8.1% overall missingness) caused by occasional sensor dropouts or communication faults, making data cleaning and masking/imputation essential.

#### Description of Each Key Function / Component for Dataset 1 (METR-LA)

* **`load_metr_la_raw(h5_path, adj_path)`:** Loads the HDF5 traffic speed time-series matrix (`metr-la.h5`) indexed by timestamps and sensor IDs, and deserializes the pickled adjacency matrix (`adj_mx_metr_la.pkl`) containing sensor ID mappings and edge weights.

* **`load_sensor_locations_and_distances(loc_csv, dist_csv)`:** Reads sensor latitude/longitude coordinates and pairwise road distance tables to map graph nodes onto physical roadway segments and compute spatial proximity weights.

* **`clean_and_impute_metr_la(speed_df, method)`:** Detects invalid/zero or missing sensor entries (`NaN` or `0.0` speed anomalies), applies temporal interpolation or historical profile imputation, and records validity masks so corrupted readings do not distort training gradients.

* **`build_spatiotemporal_windows(data_array, seq_len, horizon, feature_cols)`:** Slices the continuous 5-minute time series into overlapping input-output tensors of shape `(num_samples, num_nodes, seq_len, num_features)` and target speed arrays of shape `(num_samples, num_nodes, horizon)`.

* **`split_chronological_metr_la(X, Y, train_ratio, val_ratio)`:** Splits windows strictly in chronological order (typically 70% train, 10% validation, 20% test) to prevent temporal data leakage across overlapping windows.

#### Purpose of Dataset 1 (METR-LA)

In this project, **METR-LA** serves as the primary benchmark for evaluating how well the ST-GNN captures **high-variability urban highway dynamics, frequent congestion bottlenecks, and noisy sensor conditions**. Because Los Angeles highways experience heavy recurrent peak congestion as well as irregular slowdowns and sensor missingness, `METR-LA` tests the model's spatial diffusion capability and robustness when providing early congestion warnings for intelligent driving assistance.

### Dataset 2 (PEMS-BAY)

* **Dataset Name:** `PEMS-BAY` (California Caltrans PeMS San Francisco Bay Area Traffic Speed Dataset)

* **Source / Reference Links:**

  * PyTorch Geometric Temporal Loader & Reference: <https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/dataset/pems_bay.py>

  * Hugging Face Dataset Repository (`witgaw/PEMS-BAY`): <https://huggingface.co/datasets/witgaw/PEMS-BAY>

  * Caltrans PeMS Clearinghouse Portal: <https://pems.dot.ca.gov/>

#### Dataset Description (PEMS-BAY)

`PEMS-BAY` is collected from the California Department of Transportation (Caltrans) Performance Measurement System (PeMS) across the **San Francisco Bay Area**. It comprises **325 traffic sensors** sampled at **5-minute intervals** over six months from **January 1, 2017 to May 31, 2017** (over 52,000 time steps per sensor). Compared to `METR-LA`, `PEMS-BAY` offers a larger graph scale (325 nodes vs. 207 nodes), a longer six-month continuous span, and higher overall data completeness, making it ideal for studying large-scale corridor flow, multi-branch highway interchanges, and longer-range seasonal/weekly patterns.

#### Description of Each Key Function / Component for Dataset 2 (PEMS-BAY)

* **`load_pems_bay_data(data_dir, adj_path)`:** Loads the Bay Area speed time-series tensor (`pems-bay.h5` / `.npy` / `.parquet` splits) across all 325 PeMS stations along with the corresponding Bay Area adjacency matrix (`adj_mx_bay.pkl`).

* **`extract_temporal_covariates(timestamps)`:** Generates normalized cyclical time features—such as **time-of-day** ($\in [0, 1]$ or sine/cosine encodings) and **day-of-week** indicators—aligned with each 5-minute timestamp to help the model distinguish peak commute hours from off-peak periods.

* **`normalize_features_fit_train(train_data, val_data, test_data)`:** Computes channel-wise mean and standard deviation strictly on the training split to apply zero-mean, unit-variance standardization without leaking future statistics from validation or test sets.

* **`create_graph_signal_batches(X, Y, edge_index, edge_weight, batch_size)`:** Packages node features, multi-horizon targets, and graph edge structures into mini-batches compatible with GPU-accelerated graph neural network layers.

* **`inverse_scale_predictions(y_pred_norm, mean, std)`:** Converts normalized model outputs back into physical speed units (miles per hour, mph) for error computation and ADAS speed-advisory generation.

#### Purpose of Dataset 2 (PEMS-BAY)

In this project, **PEMS-BAY** is used to evaluate **scalability, generalization across a larger 325-sensor regional road graph, and multi-horizon stability** over a six-month span. Comparing model behavior across `PEMS-BAY` and `METR-LA` allows us to verify whether learned spatio-temporal representations transfer effectively across different urban freeway topologies and support consistent intelligent driving assistance advisories.

## Methodology

### Graph Formulation & Adjacency Matrix

We represent the roadway sensor network as a weighted graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$, where:

* $\mathcal{V}$ is the set of $N$ traffic sensors ($\vert{}\mathcal{V}\vert{} = 207$ for METR-LA; $\vert{}\mathcal{V}\vert{} = 325$ for PEMS-BAY).

* $\mathcal{E}$ is the set of directed edges capturing spatial proximity and traffic interaction between sensors.

* $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the weighted adjacency matrix constructed using a thresholded Gaussian kernel over pairwise road network distances $\text{dist}(v_i, v_j)$:

$$
\boxed{A_{ij} = \begin{cases} \exp\!\left(-\dfrac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right), & \text{if } \exp\!\left(-\dfrac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right) \ge \tau \\ 0, & \text{otherwise} \end{cases}}
$$

where $\sigma$ controls the spatial decay scale of road distance and $\tau$ is a sparsity threshold that filters out weak, distant connections so message passing remains focused on relevant neighboring roadway segments.

### Spatial Graph Convolution Module

To propagate traffic speed/flow information across connected sensors at each time step, spatial graph operations aggregate signals from neighboring nodes weighted by graph connectivity. In a normalized or diffusion-style graph aggregation layer, the hidden state $\mathbf{H}^{(l)} \in \mathbb{R}^{N \times d_l}$ at layer $l$ is updated across spatial neighborhoods as:

$$
\boxed{\mathbf{H}_{\text{spatial}}^{(l)} = \sigma_{\text{act}}\!\left(\sum_{k=0}^{K} \tilde{\mathbf{A}}_k \mathbf{H}^{(l)} \mathbf{W}_k^{(l)} + \mathbf{b}^{(l)}\right)}
$$

where $\tilde{\mathbf{A}}_k$ denotes the $k$-step forward/backward normalized transition matrices derived from $\mathbf{A}$ (capturing both downstream flow propagation and upstream congestion spillback), $\mathbf{W}_k^{(l)}$ are learnable projection weights, $\mathbf{b}^{(l)}$ is a bias term, and $\sigma_{\text{act}}(\cdot)$ is a non-linear activation function.

### Temporal Sequence Modeling Module

Traffic dynamics evolve continuously over time as vehicles accelerate, merge, and queue. After spatial aggregation, temporal sequence blocks model dependencies across historical 5-minute intervals $t - T_p + 1, \dots, t$. Using a gated temporal formulation (such as a GRU/gated temporal block), update and reset gates regulate how past states and new spatial-temporal inputs are retained:

$$
\boxed{\mathbf{z}_t = \sigma\!\left(\mathbf{W}_z [\mathbf{X}_t, \mathbf{H}_{t-1}] + \mathbf{b}_z\right), \quad \mathbf{r}_t = \sigma\!\left(\mathbf{W}_r [\mathbf{X}_t, \mathbf{H}_{t-1}] + \mathbf{b}_r\right)}
$$

$$
\boxed{\tilde{\mathbf{H}}_t = \tanh\!\left(\mathbf{W}_h [\mathbf{X}_t, \mathbf{r}_t \odot \mathbf{H}_{t-1}] + \mathbf{b}_h\right), \quad \mathbf{H}_t = (1 - \mathbf{z}_t) \odot \mathbf{H}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{H}}_t}
$$

where $\sigma(\cdot)$ is the logistic sigmoid function, $\odot$ denotes element-wise (Hadamard) multiplication, $\mathbf{z}_t$ is the update gate, $\mathbf{r}_t$ is the reset gate, and $\mathbf{H}_t$ is the updated hidden representation.

### Intelligent Driving Assistance (ADAS) Decision Layer

A key contribution of this project is connecting multi-horizon traffic predictions $\hat{V}_{i, t+\Delta t}$ (predicted speed at sensor/segment $i$ at future step $\Delta t \in \{1, \dots, T_f\}$) to **Intelligent Driving Assistance** cues along an ego vehicle's upcoming corridor $\mathcal{P} = (i_1, i_2, \dots, i_m)$:

1. **Spatial Speed Gradient & Shockwave Risk Index (**$S_{\text{shock}}$**):** Measures abrupt speed drops between consecutive segments ahead of the vehicle:

   $$
   \boxed{S_{\text{shock}}(t + \Delta t) = \max_{k \in \mathcal{P}} \frac{\max\!\left(0,\, \hat{V}_{i_k, t+\Delta t} - \hat{V}_{i_{k+1}, t+\Delta t}\right)}{\Delta x_{k, k+1}}}
   $$

   High values indicate an approaching bottleneck or stop-and-go wave where downstream traffic is substantially slower than upstream traffic.

2. **Recommended Smooth / Eco Speed Advisory (**$V_{\text{adv}}$**):** Computes a proactive target speed that dampens harsh acceleration-braking cycles by blending posted/free-flow reference speed $V_{\text{ref}}$ with predicted downstream congestion speeds weighted by proximity:

   $$
   \boxed{V_{\text{adv}}(t + \Delta t) = \min\!\left(V_{\text{limit}},\, \sum_{k=1}^{m} w_k \, \hat{V}_{i_k, t+\Delta t}\right), \quad \text{with } \sum_{k=1}^{m} w_k = 1}
   $$

   When congestion is forecast ahead, early gentle deceleration helps the vehicle arrive smoothly without sudden braking, improving ride comfort, fuel/energy efficiency, and string stability.

3. **ADAS Alert & Guidance Mapping:** Translates $S_{\text{shock}}$ and $V_{\text{adv}}$ into discrete assistance states:

   * **Normal / Free-Flow Cruise:** Maintain target cruising speed with standard adaptive cruise headway.

   * **Proactive Speed Reduction / Eco-Glide Advisory:** Begin gradual speed adjustment ahead of a predicted slowdown.

   * **Congestion / Shockwave Ahead Warning:** Increase following gap, prepare for braking, or evaluate alternative corridor routes if severe multi-horizon delay is forecasted.

### Training Objective & Optimization

Model parameters $\Theta$ are trained end-to-end by minimizing the **Mean Absolute Error (MAE)** (L1 loss) between predicted and ground-truth speeds across all $N$ sensors and $T_f$ future horizons, using valid non-missing target masks $\mathbf{M}$:

$$
\boxed{\mathcal{L}_{\text{MAE}}(\Theta) = \frac{1}{\vert{}\Omega\vert{}} \sum_{(i, \tau) \in \Omega} \left\vert{} \hat{Y}_{i, \tau} - Y_{i, \tau} \right\vert{}}
$$

where $\Omega$ indexes valid sensor-horizon pairs, optimized via Adam with learning-rate scheduling and early stopping on validation error.

## Experimental Setup & Evaluation Metrics

* **Input Window (**$T_p$**):** 12 time steps (60 minutes of historical 5-minute observations).

* **Forecast Horizons (**$T_f$**):** Evaluated across short-, medium-, and long-term horizons—typically **15 minutes** (3 steps), **30 minutes** (6 steps), and **60 minutes** (12 steps).

* **Data Split:** Chronological split into **70% Training**, **10% Validation**, and **20% Testing**.

* **Evaluation Metrics:**

  * **Mean Absolute Error (MAE):** Measures average absolute speed prediction error (in mph).

  * **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more heavily, highlighting performance during sharp congestion transitions.

  * **Mean Absolute Percentage Error (MAPE):** Measures relative percentage error across varying speed regimes.

  * **ADAS Operational Metrics:** Evaluated via shockwave detection lead time, speed-profile smoothness (reduction in sharp deceleration spikes), and advisory responsiveness ahead of congestion onset.

## Discussion

Applying Spatio-Temporal Graph Neural Networks to urban traffic and intelligent driving assistance highlights several important practical insights:

1. **Why Graph Structure Matters:** Purely temporal models treat each highway sensor independently, missing how congestion on a merging ramp or upstream artery propagates to neighboring sensors. Graph-based message passing allows the model to infer that a slowdown at one node increases the likelihood of downstream/upstream impacts along connected road segments.

2. **Horizon Trade-Offs:** Short-term forecasts (15 minutes) closely track current momentum and are especially valuable for immediate speed smoothing and headway preparation, whereas 30- to 60-minute forecasts capture broader rush-hour build-up and dissipation useful for corridor-level planning.

3. **Sensor Noise & Missing Data:** Real-world loop detectors suffer from occasional dropouts and calibration noise (particularly visible in `METR-LA`). Robust preprocessing, masking, and temporal smoothing are critical so false sensor readings do not trigger erroneous ADAS warnings.

4. **Complementarity with Onboard Vehicle Sensors:** Traffic graph forecasts do not replace local perception (cameras, radar, LiDAR, V2X); instead, they act as a **lookahead contextual layer** that gives intelligent driving systems advance notice of network-level traffic waves beyond line-of-sight range.

## Summary

This project builds a **Spatio-Temporal Graph Neural Network (ST-GNN)** framework for multi-horizon urban traffic forecasting and **Intelligent Driving Assistance**, validated on the **METR-LA** (207 sensors) and **PEMS-BAY** (325 sensors) datasets. The pipeline constructs a distance-based road network graph, models spatial interactions and temporal sequences jointly, predicts future traffic speeds across 5-minute intervals up to one hour ahead, and translates those predictions into shockwave alerts, smooth speed advisories, and driving assistance cues.

## Conclusions & Future Work

### Conclusions

* Spatio-temporal graph neural networks effectively capture coupled spatial and temporal traffic patterns across irregular roadway networks that conventional Euclidean or isolated time-series models miss.

* Evaluating on both **METR-LA** and **PEMS-BAY** demonstrates that graph-based forecasting scales across sensor networks and provides meaningful multi-step speed visibility.

* Linking traffic predictions to **intelligent driving assistance** bridges macro-level traffic analytics and micro-level driving support, enabling smoother speed adaptation and earlier awareness of traffic bottlenecks.

### Future Work

* **Dynamic & Attention-Based Graphs:** Incorporate adaptive or attention-driven adjacency matrices that evolve dynamically with real-time congestion states and weather/incident events.

* **Multi-Modal & V2X Integration:** Fuse loop-detector telemetry with incident feeds, weather data, and Vehicle-to-Everything (V2X) / connected-vehicle trajectories.

* **Closed-Loop Driving Simulation:** Test the ADAS advisory layer inside microscopic traffic and driving simulators (such as SUMO or CARLA) to quantify fuel/energy savings, travel-time reliability, and braking smoothness under realistic mixed-autonomy traffic.

## Repository Structure & Usage

```
stgnn-traffic-adas/
├── data/
│   ├── metr-la/
│   │   ├── metr-la.h5
│   │   ├── adj_mx_metr_la.pkl
│   │   └── graph_sensor_locations_metr_la.csv
│   └── pems-bay/
│       ├── pems-bay.h5
│       ├── adj_mx_bay.pkl
│       └── graph_sensor_locations_bay.csv
├── src/
│   ├── data_loader.py          # Dataset loading, imputation, normalization & windowing
│   ├── graph_utils.py          # Adjacency matrix construction & graph normalization
│   ├── model_stgnn.py          # Spatio-Temporal GNN architecture & training loop
│   ├── adas_module.py          # Shockwave detection, speed advisory & assistance logic
│   └── evaluate.py             # MAE, RMSE, MAPE & horizon/ADAS evaluation scripts
├── train.py                    # Main training entry point
├── inference_adas.py           # Run multi-horizon forecast & ADAS advisories
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```

### Quick Start Commands

```
# 1. Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train ST-GNN on METR-LA or PEMS-BAY
python train.py --dataset metr-la --seq_len 12 --horizon 12 --epochs 50
python train.py --dataset pems-bay --seq_len 12 --horizon 12 --epochs 50

# 4. Evaluate forecasts and generate Intelligent Driving Assistance advisories
python inference_adas.py --dataset metr-la --checkpoint checkpoints/best_model.pt

```

## Author

* **Name:** Muhammad Usman Akram

* **LinkedIn:** <https://www.linkedin.com/in/muhammad-usman-akram-b29685251/>

* **GitHub:** <https://github.com/engrusman00109?tab=repositories>

* **Email:** <mailto:musman00109@gmail.com>