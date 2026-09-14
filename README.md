# NeuTraL AD — Thyroid Dataset Reproduction

This repository contains a PyTorch reproduction of **NeuTraL AD** (Qiu et al., "Self-Supervised Anomaly Detection With Neural Transformations," IEEE TPAMI, 
Vol. 47, No. 3, March 2025), which learns a set of neural transformations and an encoder jointly via a Deterministic Contrastive Loss (DCL) that also serves 
as the anomaly score, evaluated here on the Thyroid tabular anomaly detection benchmark, achieving a test AUC of 0.8555 using 11 learnable transformations, 
a residual transformation mode, and early stopping on validation AUC.
