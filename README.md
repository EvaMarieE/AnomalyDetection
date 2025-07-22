# Anomaly Detection in Chest X-Rays using GLASS and MAD-AD

This repository contains code and experiments for unsupervised anomaly detection in chest X-ray images using two state-of-the-art methods:
- **GLASS** (Generative Latent Anomaly Score)
- **MAD-AD** (Multiresolution Anomaly Detection via Feature Distillation)

The goal is to identify anomalies (e.g. disease indicators) in chest X-rays without requiring annotated pathology data during training. Both methods are evaluated on chest X-ray datasets with a focus on medical relevance and performance under data scarcity.

## Methods Implemented

# GLASS (ECCV 2022)
GLASS leverages latent space density estimation and generative models to detect anomalies based on reconstruction error and feature likelihoods.

# MAD-AD (CVPR 2023)
MAD-AD distills normal features using multiresolution image patches and deep feature comparison to highlight out-of-distribution regions.


- GLASS  Original -> https://github.com/cqylunlun/GLASS
- MAD-AD Original -> https://github.com/farzad-bz/MAD-AD
