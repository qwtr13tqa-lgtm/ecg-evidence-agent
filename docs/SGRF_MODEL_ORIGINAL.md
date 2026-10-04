<h1 align="center">SGRF-Net: Shape-Guided Residual Fusion Network for 12-Lead ECG Anomaly Detection</h1>

<p align="center">
<p align="center">
<a href="#"><strong>Jiaming Liu</strong></a>
·
<a href="#"><strong>Jinqing Qi</strong></a>
</p>

<h4 align="center">Published in <a href="#">ICSIP 2026</a></h4>
</p>

Abstract

Electrocardiogram (ECG) anomaly detection plays a vital role in cardiovascular health monitoring and intelligent diagnosis. High-quality morphological features are critical for reliable analysis of 12-lead ECGs; however, signal quality often varies significantly across different leads, and individual leads may be severely corrupted by noise or electrode-related artifacts. To alleviate this problem, we propose the Shape-Guided Residual Fusion Network (SGRF-Net).

To effectively capture lead-specific morphological features, a Morphology-Driven Gating (MDG) module (implemented as Shapelet-Guided Gating) is designed. MDG provides a scale-insensitive metric that dynamically reweights features based on consistency with normal priors. This module constrains the reconstruction to the normal manifold, preserving normal rhythms while inducing a morphological mismatch for anomalous inputs, thereby amplifying errors to enhance discriminability. Extensive experiments on the large-scale PTB-XL dataset demonstrate the effectiveness of SGRF-Net for robust ECG anomaly detection under conditions of varying signal quality across leads.

## Prerequisites

<ul>
<li>Pytorch</li>
<li>Torchvision</li>
<li>Numpy</li>
<li>SciPy</li>
<li>HeartPy</li>
<li>Scikit-learn</li>
<li>Matplotlib</li>
<li>tqdm</li>
</ul>

## Datasets

To validate the effectiveness of our model, we conduct the experiments benchmark in the PTB-XL dataset. We follow the standard dataset preprocessing as in Jiang et al. Please ensure your data is processed and stored in .npy format within the data/Processed_PTBXL/ directory.

Usage

## Training

<ul>
<li>To train the SGRF-Net with the shape-guided gating mechanism:</li>

python train.py


<li><strong>Key Arguments:</strong></li>
<ul>
<li><code>--shapex_num_shapelets</code>: Number of learnable shapelets (default: 16).</li>
<li><code>--mask_ratio_time</code>: Ratio of time-domain masking (default: 30).</li>
<li><code>--mask_ratio_spec</code>: Ratio of spectrogram masking (default: 20).</li>
<li><code>--lambda_match</code> / <code>--lambda_div</code>: Weights for shapelet matching and diversity losses.</li>
<li><code>--mask_loss</code>: Enable peak-based error calculation during training/testing.</li>
</ul>
</ul>

## Testing & Evaluation

<ul>
<li>To evaluate the model performance and calculate metrics (AUC, F1, etc.):</li>

python test1.py


<li>To visualize the latent space using t-SNE:</li>

python test-SNE.py


</ul>

## Experimental Results

The experimental results on the PTB-XL dataset demonstrate the superiority of SGRF-Net compared to restoration-based baselines:

| Model | AUC | F1-Score  | Precision | Recall    |
|-------|-----|-----------|-----------|-----------|
| Baseline (TSRNet) | 0.860 | 0.798     | 0.853     | 0.751     |
| **SGRF-Net (Ours)** | **0.887** | **0.826** | **0.871** | **0.785** |

## Citation

@article{sgrfnet2026,
  title={SGRF-Net: Shape-Guided Residual Fusion Network for 12-Lead ECG Anomaly Detection},
  author={Jiaming Liu and Jinqing Qi},
  journal={ICSIP},
  year={2026}
}


## Acknowledgment

A significant part of this code is adapted from and inspired by these previous works:

<ul>
<li><strong>TSRNet</strong>: Bui et al. (ISBI 2024)</li>
<li><a href="https://github.com/MediaBrain-SJTU/ECGAD">Jiang et al.</a></li>
<li><a href="https://github.com/UARK-AICV/ECG_SSL_12Lead">Phan et al.</a></li>
</ul>

