```markdown
# Multi-Granularity Anchor Embedded Discriminative Latent Low-Rank Fuzzy Representation Clustering

This repository provides the MATLAB implementation of:

> **Multi-granularity anchor embedded discriminative latent low-rank fuzzy representation clustering for color image segmentation**

The proposed method is designed for unsupervised color image segmentation by integrating multi-granularity anchors, discriminative latent low-rank representation, fuzzy clustering, and graph-based spatial information.

---

## Overview

Existing low-rank representation-based clustering methods may suffer from excessive singular-value shrinkage, loss of discriminative information, and over-smoothed segmentation boundaries.

To address these issues, this work introduces a multi-granularity anchor embedded fuzzy representation clustering framework with the following components:

- **Multi-granularity anchor construction** for representing image data at different scales;
- **Anchor-based similarity learning** to reduce computational cost;
- **Discriminative latent low-rank representation** for modeling the underlying subspace structure;
- **Hyperbolic tangent rank regularization** to alleviate excessive singular-value shrinkage;
- **Fuzzy normalization** for obtaining soft cluster assignments;
- **Local and non-local image information** for preserving spatial structures and segmentation details.

The final segmentation result is obtained from the learned fuzzy membership matrix.

---

## Method Pipeline

The general processing pipeline is:

```text
Input color image
        │
        ▼
Color-space conversion and feature extraction
        │
        ▼
Local / non-local information construction
        │
        ▼
Multi-granularity anchor generation
        │
        ▼
Anchor similarity and adjacency construction
        │
        ▼
Low-rank fuzzy representation optimization
        │
        ▼
Fuzzy membership normalization
        │
        ▼
Cluster-label generation and image segmentation
```

---

## Repository Structure

```text
.
├── README.md
├── demo_to_run.m
├── Label_image.m
│
├── adm m_optimization.m
├── build_anchor_adjacency_func.m
├── compute_anchor_similarity_func.m
├── compute_tanh_rank_func.m
├── fuzzy_normalize_func.m
├── fuzzy_normalize_gradient_func.m
├── generate_coarse_anchors_func.m
├── local_variance.m
├── non_local_information.m
└── colorspace.m
```

### Main Scripts

| File | Description |
|---|---|
| `demo_to_run.m` | Main demonstration script for running the proposed method |
| `Label_image.m` | Converts the clustering or membership result into an image-label map |

### Core Functions

| File | Description |
|---|---|
| `generate_coarse_anchors_func.m` | Generates coarse-grained anchors |
| `compute_anchor_similarity_func.m` | Computes similarities between image samples and anchors |
| `build_anchor_adjacency_func.m` | Constructs the anchor adjacency relationship |
| `compute_tanh_rank_func.m` | Computes the hyperbolic tangent rank regularization |
| `fuzzy_normalize_func.m` | Performs fuzzy membership normalization |
| `fuzzy_normalize_gradient_func.m` | Computes the gradient associated with fuzzy normalization |
| `admm_optimization.m` | Solves the optimization problem using the ADMM framework |
| `local_variance.m` | Extracts local variance information |
| `non_local_information.m` | Computes non-local image information |
| `colorspace.m` | Performs color-space transformation or color feature processing |

---

## Requirements

- MATLAB
- MATLAB Image Processing Toolbox is recommended
- Sufficient memory for loading and processing the input image

The code is implemented using MATLAB `.m` files and does not require Python.

---

## Getting Started

Clone or download this repository:

```bash
git clone https://github.com/gocchong/MGA-Subspace-clustering.git
cd MGA-Subspace-clustering
```

Open MATLAB, set the repository directory as the current folder, and run:

```matlab
demo_to_run
```

The demo script demonstrates the complete processing procedure, including feature construction, anchor generation, optimization, fuzzy membership estimation, and segmentation-label generation.

---

## Input Data

The demo can be adapted to different color images by modifying the image-loading and parameter settings in:

```matlab
demo_to_run.m
```

For example:

```matlab
img = imread('your_image.jpg');
```

The input should generally be a color image with three channels. If a different image format or feature representation is used, the corresponding preprocessing section should be adjusted accordingly.

---

## Output

The algorithm produces fuzzy membership maps and segmentation-label results.

Typical output files include:

```text
Cluster1_Membership.png
Cluster2_Membership.png
Cluster3_Membership.png
cluster_mask.png
```

The membership maps visualize the degree to which each pixel belongs to a specific cluster. The final label image can be generated using:

```matlab
Label_image
```

Example visualization files included in the repository are:

- `Cluster1_Membership.png`
- `Cluster2_Membership.png`
- `Cluster3_Membership.png`
- `cluster_mask.png`
- `3096.jpg`
- `h2.jpg`
- `h5.jpg`
- `s11.jpg`

---

## Important Parameters

The main algorithmic parameters can be adjusted in `demo_to_run.m`, including:

- Number of clusters;
- Number and scale of anchors;
- Fuzzy normalization parameters;
- Low-rank regularization parameters;
- ADMM optimization parameters;
- Stopping tolerance and maximum iteration number.

For different images, the number of clusters and anchor-related parameters may need to be adjusted according to the image content and expected segmentation classes.

---

## Implementation Notes

The optimization procedure is implemented using the alternating direction method of multipliers (ADMM). The main optimization process is contained in:

```matlab
admm_optimization.m
```

The method uses multiple anchor scales to balance:

- Global structural information;
- Local image details;
- Computational efficiency;
- Robust fuzzy cluster assignment.

The hyperbolic tangent rank function is implemented in:

```matlab
compute_tanh_rank_func.m
```

---

## Reproducibility

To reproduce the demo results:

1. Download or clone this repository;
2. Open MATLAB and set the repository as the current directory;
3. Check the image paths and parameters in `demo_to_run.m`;
4. Run:

```matlab
demo_to_run
```

5. Use `Label_image.m` to generate or visualize the final segmentation labels if necessary.

Because the optimization result may depend on image size, parameter settings, and initialization, the results may exhibit minor differences across MATLAB versions or computational environments.

---

## Citation

If you find this code or paper useful, please cite:

```bibtex
@article{chong2027made_lfrc,
  title   = {Multi-granularity anchor embedded discriminative latent low-rank fuzzy representation clustering for color image segmentation},
  author  = {Chong, Qianpeng and Wen, Jiakun and Wei, Guangyi and Ma, Rong and Bao, Rui and Long, Yao and Zheng, Xin and Zeng, Weny},
  journal = {Information Sciences},
  volume  = {759},
  pages   = {124029},
  year    = {2027},
  publisher = {Elsevier},
  doi     = {10.1016/j.ins.2026.124029}
}
```

---

## License

Please refer to the license information provided in this repository.

The code and example images should be used in accordance with their respective licenses.

---

## Contact

For questions or suggestions, please open an issue in this repository or contact the authors.

- **Repository:** [https://github.com/gocchong/MGA-Subspace-clustering](https://github.com/gocchong/MGA-Subspace-clustering)
```
