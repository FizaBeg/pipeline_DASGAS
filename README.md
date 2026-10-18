# Unsupervised Autoencoder with Monte Carlo Dropout for Epistemic Uncertainty Quantification for Spatiotemporal Signal Analysis

This repository contains the production code and interactive visualization modules for an unsupervised, topology-aware deep learning pipeline designed to monitor spatiotemporal fiber-optic Distributed Acoustic Sensing (DAS) data streams.

## 📂 Core Repository Files

### 1. `run_pipeline_master.py`
A standalone production execution script that automates the complete 12-stage pipeline sequentially:
- **Data Conditioning & Invariant Features:** Implements spatial common-mode median filtering, temporal Discrete Wavelet Transforms (1D DWT via `db4`), and 5D statistical moment extraction (RMS, Variance, Skewness, Kurtosis, Peak-to-Peak) over overlapping windows.
- **Probabilistic Autoencoder (Statistical Branch):** Trains an unsupervised neural network bottleneck using 30 stochastic Monte Carlo (MC) Dropout inference passes to isolate mean reconstruction anomalies (MSE) and compile epistemic model variance metrics.
- **Graph Laplacian Regularization (Physical Branch):** Constructs a spatiotemporal graph network over window centers and executes an implicit membrane diffusion process via the combinatorial Graph Laplacian operator to regularize pixel noise according to the physical cable geometry.
- **Evaluation:** Automatically compiles and exports the final Signal-to-Noise Ratio (SNR) and Jensen-Shannon Divergence (JSD) performance metrics table.

### 2. `Map_Rendering_Module.ipynb`
An interactive Jupyter Notebook workspace that maps and displays the evaluated output arrays from disk:
- Generates high-resolution **Autoencoder Reconstruction Error Maps** (Mean MSE).
- Renders **Monte Carlo Uncertainty Quantification Profiles** (Model Epistemic Variance).
- Plots **Graph Laplacian Geometrically Diffused Heatmaps** tracking continuous fluid plume boundaries.

---

## ⚙️ Core Dependencies & Setup
This framework was compiled within an isolated Miniconda node running Python 3.10. Install core packages via:
```bash
pip install numpy scipy scikit-learn pywavelets torch matplotlib h5py tqdm
```

---

## 📊 Pipeline Performance Matrix
The core dispatcher prints the following verified experimental benchmarks upon execution termination:

| Pipeline Processing Stage | Cum. SNR | Incremental $\Delta$SNR | Separability (JSD) |
| :--- | :---: | :---: | :---: |
| Stage 01: Raw Matrix Input | -1.99 dB | — | 0.999 |
| Stage 03: Common-Mode Filter | -1.99 dB | +0.00 dB | 0.999 |
| Stage 04: Wavelet Denoising | -7.21 dB | -5.22 dB | 0.999 |
| Stage 09: Statistical Autoencoder | **22.68 dB** | **+29.89 dB** | 0.018 |
| Stage 10: Graph Laplacian | 22.68 dB | +-0.00 dB | **0.153 ( Tenfold Increase)** |

---

## 🎓 Project Acknowledgments
This research was conducted at the Department of Computing, **Universiti Teknologi PETRONAS (UTP)**, and was generously funded under the **TOTALENERGIES EP Malaysia Research Grant** for the project titled *"Data Analytics and Visualization Development System for Subsurface CO₂ Storage and Fluid Production."* Open-source dataset arrays were provided by the transnational **DASGAS** project team at the **ECCSEL Svelvik CO₂ Field Lab** testing facility.
