# Short Video Streaming

## Introduction

This repository contains the implementation and experimental code for **short-video streaming and adaptive bitrate (ABR)** research.

The repository provides a common short-video streaming simulator together with implementations of several short-video ABR algorithms, including **PDAS, Incendio, Dashlet, DeLoad, PBR-ARD, and SecBAD(PARA)**. It also contains scripts for dataset preparation, network-trace processing, experiment execution, runtime/memory measurement, and result analysis.

The repository is mainly organized into three parts:

- **Algorithm implementations:** Implementations and evaluation scripts for multiple short-video ABR algorithms.
- **Simulation environment:** A common simulator for modeling video playback, user viewing behavior, network conditions, buffering, rebuffering, and bandwidth waste.
- **Data processing and analysis:** Scripts and notebooks for preparing datasets, processing network traces, measuring algorithm performance, and analyzing experimental results.

---

# Code Structure

The repository is organized as follows:

```text
short_video/
│
├── config_algorithm.py
├── requirements.txt
├── README.md
│
├── simulator/
│   ├── controller.py
│   ├── controller_back_up.py
│   ├── mpc_module.py
│   ├── network_module.py
│   ├── short_video_load_trace.py
│   ├── user_module.py
│   └── video_player.py
│
├── algorithms/
    │
    ├── PBR_ARD/
    │   ├── simulator_ARD/
    │   ├── config_algorithm.py
    │   ├── solution_ard.py
    │   ├── run_ard.py
    │   └── test_one_dataset_ard.py
    │
    ├── dashlet/
    │   ├── solution_dashlet.py
    │   ├── run_dashlet.py
    │   └── test_one_dataset_dashlet.py
    │
    ├── deload/
    │   ├── model/
    │   ├── simulator_deload/
    │   ├── train.py
    │   ├── solution_deload.py
    │   ├── run_deload.py
    │   └── test_one_dataset_deload.py
    │
    ├── incendio/
    │   ├── Incendio_model.py
    │   ├── train_IL.py
    │   ├── train_RL.py
    │   ├── solution_Incendio.py
    │   ├── run_incendio.py
    │   └── test_one_dataset_incendio.py
    │
    ├── pdas/
    │   ├── Training.py
    │   ├── solution_pdas.py
    │   ├── run_pdas.py
    │   └── test_one_dataset_pdas.py
    │
    ├── secbad/
    │   ├── algorithms/
    │   ├── config/
    │   ├── environments/
    │   ├── models/
    │   ├── main.py
    │   ├── mixed_learner.py
    │   ├── vae.py
    │   └── test_one_dataset_secbad.py
    │
    ├── data/
    │   ├── generate_dataset_2s.py
    │   ├── sample_user_by_KuaiRand_view_time.py
    │   ├── count_view_percentage.py
    │   ├── analyze_network_trace.py
    │   └── network_traces/
    │
    ├── analyze_network.ipynb
    ├── get_video_duration_video_ret.ipynb
    └── plot_mesurement_study.ipynb
```

---

# Algorithms

The `algorithms/` directory contains implementations and evaluation scripts for multiple short-video ABR algorithms.

## PDAS

The `algorithms/pdas/` directory contains the PDAS implementation.

- `Training.py`: Training of fastmpc.
- `MPC_balanced.txt`: Trained fastmpc.
- `solution_pdas.py`: PDAS decision algorithm.
- `run_pdas.py`: Runs the PDAS simulation.
- `test_one_dataset_pdas.py`: Runs PDAS on a configured dataset and network-trace set.

Run the configured test:

```bash
cd algorithms/pdas
python test_one_dataset_pdas.py
```

---

## Incendio

The `algorithms/incendio/` directory contains the Incendio implementation and training code.

- `Incendio_model.py`: Incendio model definition.
- `train_IL.py`: Imitation-learning training code.
- `train_RL.py`: Reinforcement-learning training code.
- `replay_buffer.py`: Replay buffer implementation.
- `exp_dataset.py`: Experience-dataset related code.
- `solution_Incendio.py`: Incendio decision algorithm.
- `run_incendio.py`: Runs the Incendio simulation.
- `test_one_dataset_incendio.py`: Runs Incendio on the configured dataset and network traces.

Run the configured test:

```bash
cd algorithms/incendio
python test_one_dataset_incendio.py
```

---

## Dashlet

The `algorithms/dashlet/` directory contains the Dashlet implementation.

- `Training.py`: Training of fastmpc.
- `MPC_balanced.txt`: Trained fastmpc.
- `solution_dashlet.py`: Dashlet decision algorithm.
- `run_dashlet.py`: Runs the Dashlet simulation.
- `test_one_dataset_dashlet.py`: Runs Dashlet on the configured dataset and network traces.

Run the configured test:

```bash
cd algorithms/dashlet
python test_one_dataset_dashlet.py
```

---

## DeLoad

The `algorithms/deload/` directory contains the DeLoad implementation.

- `model/PPO.py`: PPO-related model implementation.
- `train.py`: Training code.
- `exp_buffer.py`: Experience buffer.
- `Weibull.py`: Weibull-related functionality.
- `solution_deload.py`: DeLoad decision algorithm.
- `run_deload.py`: Runs the DeLoad simulation.
- `test_one_dataset_deload.py`: Runs DeLoad on the configured dataset and network traces.
- `simulator_deload/`: DeLoad-specific simulator components.

Run the configured test:

```bash
cd algorithms/deload
python test_one_dataset_deload.py
```

---

## PBR-ARD

The `algorithms/PBR_ARD/` directory contains the PBR-ARD implementation and its dedicated simulator.

- `solution_ard.py`: PBR-ARD decision algorithm.
- `run_ard.py`: Runs the PBR-ARD simulation.
- `test_one_dataset_ard.py`: Runs PBR-ARD on the configured dataset and network traces.
- `simulator_ARD/`: PBR-ARD-specific simulator components.

Run the configured test:

```bash
cd algorithms/PBR_ARD
python test_one_dataset_ard.py
```

---

## SecBAD

The `algorithms/secbad/` directory contains the SecBAD implementation.

The implementation includes:

- `algorithms/`: A2C, PPO, and related learning components.
- `config/`: Configuration files for short-video streaming.
- `environments/`: Short-video streaming environment.
- `models/`: Encoder, decoder, policy, SAC, and DQN-related models.
- `mixed_learner.py`: Main learner implementation.
- `vae.py`: VAE-related implementation.
- `main.py`: Main entry point.
- `test_one_dataset_secbad.py`: Evaluation script.

Run the configured evaluation:

```bash
cd algorithms/secbad
python test_one_dataset_secbad.py
```

---

# Short-Video Streaming Simulator

The `simulator/` directory provides the common simulation environment used by several algorithms.

## Environment

`simulator/controller.py` defines the main `Environment` class.

The environment models:

- Multiple concurrently visible short videos.
- User viewing duration and swipe behavior.
- Video playback and buffering.
- Network throughput.
- Bitrate selection.
- Rebuffering.
- Bandwidth usage and wasted bandwidth.
- Video switching between consecutive short videos.

The simulator maintains multiple `Player` instances and corresponding user-retention models. The default player configuration contains **5 concurrently managed videos**.

## Network Module

`simulator/network_module.py` implements network-trace processing and network evolution during simulation.

## Video Player

`simulator/video_player.py` implements video playback and buffer-related operations.

## User Model

`simulator/user_module.py` provides the user-retention model used to represent viewing duration and user behavior.

## Trace Loading

`simulator/short_video_load_trace.py` loads network traces used by the streaming simulator.

---

# Data Preparation

The `algorithms/data/` directory contains scripts for preparing user-behavior data and network traces.

`get_video_duration_video_ret.ipynb` samples user playback durations from KuaiRand-27K.

Run:

```bash
cd algorithms/data
python get_video_duration_video_ret.ipynb
```

It generates dataset files under:

```text
random_sampled_video/
├── video_names.csv
├── user_ret/
├── user_switch_prob/
└── view_duration/
```

## Dataset Generation

`generate_dataset_2s.py` creates a sampled dataset from the available video/user-retention data.

The script uses a fixed random seed and creates a dataset containing **100 videos**.

Run:

```bash
cd algorithms/data
python generate_dataset_2s.py
```

It generates dataset files under:

```text
dataset/
├── video_names.csv
├── user_ret/
├── user_switch_prob/
├── view_duration/
└── sample_user/
```

## User Viewing-Duration Sampling

`sample_user_by_KuaiRand_view_time.py` samples user playback durations according to specified viewing-time intervals from data extract from KuaiRand-27K.

Run:

```bash
cd algorithms/data
python sample_user_by_KuaiRand_view_time.py <interval> <begin>
```

For example:

```bash
python sample_user_by_KuaiRand_view_time.py 12 0
```

The script is designed to generate sampled user playback-duration data for experiments with different viewing-time ranges.

## Network Trace Sampling

`network_traces/sample_network_traces.py` contains code for sampling network traces from the source trace:

```text
NewFile-HighDensity-4G.txt
```

The script converts the source trace values into Mbps and can generate sampled trace files for experiments.

---

# Performance Metrics

The simulator and algorithm implementations record several streaming performance metrics, including:

- **QoE**
- **Average video quality / bitrate**
- **Quality smoothness**
- **Rebuffering**
- **Bandwidth usage**
- **Bandwidth waste**
- **Viewing duration**
- **Downloaded and viewed video chunks**

The QoE calculation uses the parameters defined in `config_algorithm.py`

---

# Result Analysis

The repository contains scripts and Jupyter notebooks for analyzing experimental results.

## Measurement Analysis

- `algorithms/plot_mesurement_study.ipynb`: Measurement-study visualization.
- `algorithms/analyze_network.ipynb`: Network-trace analysis.
- `algorithms/get_video_duration_video_ret.ipynb`: Analysis of video duration and viewing behavior.

---

# Environment Setup

## Requirements

The repository provides the following Python dependencies in `requirements.txt`:

```text
numpy==1.24.4
torch==2.4.1
gym==0.26
matplotlib
tensorboard==2.14.0
scipy==1.10.1
```

A Python environment compatible with these dependencies is recommended.

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

