# Training Dynamics and State Taxonomy in Deep Visual Recognition Networks

This repository provides the notebook and source code to train and analyze various deep visual recognition models (including ResNet, VGG, DenseNet, MobileNetV2, and ViT) on the CIFAR-10 and CIFAR-100 datasets. It applies dynamical metrics to classify the internal states and training dynamics of these networks.

## What does this code do & Key Findings?

This code calculates and analyzes dynamic metrics—such as Effective Hierarchy ($\bar{H}_{eff}$), Metastability ($\bar{M}$), and the composite index ($\bar{\Psi}$)—across different network architectures.

**Key Findings:**
* **Clear separation by dataset:** The most consistent finding across the nine configurations is the separation of $\bar{H}_{eff}$ values by dataset. 
* **CIFAR-10 vs. CIFAR-100:** The six CIFAR-10 configurations (five trained from scratch and the pretrained ViT) converge to a mean $\bar{H}_{eff}$ in the range `[0.83, 0.98]`. In contrast, the three CIFAR-100 configurations remain in the `[0.04, 0.30]` range.
* **Architectural Robustness:** This separation is robust to architectural variation. For example, ResNet-152 on CIFAR-10 achieves $\bar{H}_{eff} = 0.931$, while ResNet-50 on CIFAR-100 (a substantially deeper architecture on the same image modality) yields $\bar{H}_{eff} = 0.057$.
* **Parameter Stability:** This dataset separation is robust to $H_{opt} \in [0.6, 0.8]$ with $\sigma_H \geq 0.10$, but reverses when $H_{opt} = 0.5$. The sign pattern of the correlation $r(\Psi, acc)$ remains stable across all weight combinations $w_H \in \{0.3, 0.5, 0.7\}$ tested.

### Dynamical Metric Summary

*Table 1: Summary across all visual recognition models. $\bar{H}_{eff}$ and $\bar{\Psi}$ are epoch-means (± std). $\bar{M}$ is the mean metastability index. $r(H_z, M_z)$ reports Pearson inter-field synchrony and $r(\Psi, acc)$ reports the Pearson correlation of $\Psi$ with validation accuracy.*

| Model | Dataset | Epochs | Best acc. (%) | $\bar{H}_{eff}$ | $\bar{M}$ | $\bar{\Psi}$ | $r(H_z, M_z)$ | $r(\Psi, acc)$ | State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-18 | CIFAR-10 | 25 | 78.6 (ep. 23) | 0.830 ± 0.046 | 0.0806 | 0.000 ± 0.943 | 0.777 | -0.661 | Transitional |
| ResNet-34 | CIFAR-10 | 27 | 76.8 (ep. 26) | 0.852 ± 0.046 | 0.0822 | -0.206 ± 0.704 | 0.514 | -0.342 | Transitional |
| ResNet-152 | CIFAR-10 | 38 | 68.6 (ep. 38) | 0.931 ± 0.031 | 0.0976 | -0.745 ± 0.534 | 0.600 | -0.436 | Metastable H-I |
| DenseNet-121 | CIFAR-10 | 25 | 77.7 (ep. 23) | 0.880 ± 0.068 | 0.0664 | 0.000 ± 0.561 | -0.371 | +0.600 | Stable Convergent |
| MobileNetV2 | CIFAR-10 | 27 | 68.9 (ep. 25) | 0.951 ± 0.012 | 0.0450 | 0.000 ± 0.740 | 0.095 | +0.313 | Metastable H-I |
| ViT (pretrained) | CIFAR-10 | 30 | 89.3 (ep. 18) | 0.980 ± 0.011 | 0.0490 | 0.000 ± 0.720 | 0.036 | -0.330 | Metastable H-I |
| ResNet-50 | CIFAR-100 | 56 | 54.1 (ep. 53) | 0.057 ± 0.019 | 0.0770 | -0.114 ± 0.860 | 0.864 | -0.760 | Rigidly Sync. |
| ResNet-101 | CIFAR-100 | 58 | 49.1 (ep. 57) | 0.070 ± 0.026 | 0.0779 | -0.072 ± 0.970 | 0.885 | -0.725 | Rigidly Sync. |
| VGG-16 | CIFAR-100 | 55 | 63.8 (ep. 53) | 0.303 ± 0.006 | 0.0957 | 0.000 ± 0.798 | 0.274 | -0.396 | Partial Integr. |

## Who is this for?

* **Deep Learning Researchers** studying training dynamics, network convergence, and internal representations of neural networks.
* **AI Scientists** interested in applying concepts of metastability, hierarchy, and dynamical systems to visual recognition models.
* **Students and Practitioners** looking for practical implementations of complex network analysis on standard benchmarks like CIFAR-10/100.

## Reference

More details can be found in our paper: ...
