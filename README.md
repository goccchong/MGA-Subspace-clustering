# MGA-Subspace-clustering

```markdown
# Multi-Granularity Anchor Embedded Discriminative Latent Low-Rank Fuzzy Representation Clustering

Official implementation of:

> **Multi-granularity anchor embedded discriminative latent low-rank fuzzy representation clustering for color image segmentation**

This repository provides the implementation of **MADE-LFRC**, an unsupervised image segmentation method based on multi-granularity anchors, discriminative latent low-rank representation, and fuzzy clustering.

## Overview

Low-rank representation-based subspace clustering is effective for image segmentation because it can model high-dimensional data using multiple low-dimensional subspaces. However, existing methods may suffer from:

- Excessive shrinkage of singular values;
- Loss of intra-class variation;
- Oversmoothing of image regions;
- Insufficient preservation of local details;
- Boundary blurring caused by single-granularity anchor representations.

To address these limitations, we propose **Multi-granularity Anchor embedded Discriminative Latent low-rank Fuzzy Representation Clustering (MADE-LFRC)**.

<p align="center">
  <img src="figures/framework.png" width="850">
</p>

## Key Features

- **Discriminative latent low-rank representation**  
  Models latent subspace structures while preserving more discriminative image information.

- **Hyperbolic tangent rank regularization**  
  Reduces the excessive shrinkage of dominant singular values and suppresses minor singular values.

- **Multi-granularity anchor strategy**  
  Combines anchors from different scales to capture both global structures and local details.

- **Graph-Laplacian regularization**  
  Preserves the spatial and neighborhood relationships among image samples.

- **Fuzzy normalization**  
  Integrates multi-scale affinity matrices into a coherent fuzzy partition.

- **Unsupervised image segmentation**  
  No initial cluster-center selection or manual pixel-level annotation is required.

## Method

The proposed method jointly optimizes a multi-granularity low-rank representation and fuzzy membership matrix:

\[
\min_{\mathbf{Z},\mathbf{U},\ldots}
\mathcal{L}(\mathbf{Z},\mathbf{U},\ldots),
\]

where:

- \(\mathbf{Z}\) denotes the latent low-rank representation;
- \(\mathbf{U}\) denotes the fuzzy membership matrix;
- Multi-scale anchor matrices are used to reduce computational complexity;
- Hyperbolic tangent rank regularization preserves dominant singular values;
- Graph-Laplacian regularization maintains local spatial structures.

The final segmentation map is obtained from the learned fuzzy membership matrix.

## Installation

```bash
git clone https://github.com/gocchong/MGA-Subspace-clustering.git
cd MGA-Subspace-clustering

# Create a virtual environment
conda create -n made-lfrc python=3.9
conda activate made-lfrc

# Install dependencies
pip install -r requirements.txt
```

## Requirements

The implementation requires:

- Python 3.9+
- NumPy
- SciPy
- scikit-image
- scikit-learn
- OpenCV
- Matplotlib

The complete dependency list is provided in `requirements.txt`.

## Dataset Preparation

Place the datasets in the following directory:

```text
data/
├── natural_images/
├── remote_sensing_images/
└── ...
```

The supported datasets include:

- Natural image datasets;
- Large-scale remote-sensing image datasets.

> Please refer to the dataset descriptions in the paper and follow the corresponding dataset licenses.

## Usage

### Run the default configuration

```bash
python main.py \
    --dataset <dataset_name> \
    --num_clusters <number_of_clusters>
```

### Specify the input image

```bash
python main.py \
    --image_path path/to/image.png \
    --num_clusters 4 \
    --num_anchors 100
```

### Evaluate segmentation results

```bash
python evaluate.py \
    --result_dir results/<dataset_name>
```

> The exact command-line arguments may vary according to the final released implementation.

## Results

MADE-LFRC is evaluated on natural images and large-scale remote-sensing images. Experimental results demonstrate that the proposed method can:

- Preserve richer local image details;
- Reduce region oversmoothing;
- Improve segmentation boundary quality;
- Achieve competitive or superior performance compared with existing subspace clustering methods.

Example results:

<p align="center">
  <img src="figures/qualitative_results.png" width="850">
</p>

Quantitative results will be added after the official experimental files are released.

## Repository Structure

```text
.
├── data/                  # Dataset directory
├── figures/               # Figures and visualizations
├── models/                # Model and optimization components
├── utils/                 # Utility functions
├── main.py                # Main entry point
├── evaluate.py            # Evaluation script
├── requirements.txt       # Python dependencies
└── README.md
```

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

## Paper

- **Journal:** Information Sciences
- **DOI:** [10.1016/j.ins.2026.124029](https://doi.org/10.1016/j.ins.2026.124029)
- **Code:** [https://github.com/gocchong/MGA-Subspace-clustering](https://github.com/gocchong/MGA-Subspace-clustering)

## License

Please refer to the `LICENSE` file for the license of this repository.

The code and datasets should be used in accordance with their respective licenses.

