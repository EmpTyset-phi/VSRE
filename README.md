# VSRE Calculator for LiDAR Point Clouds

## 📌 Overview

A Python tool for calculating Vegetation Structure Relative Entropy (VSRE) from LiDAR point cloud data (.LAS files). This tool implements the methodology from the paper: "Quantifying the structural complexity of coastal vegetation based on relative entropy from multi-density LiDAR point clouds". The VSRE metric quantifies vegetation structural complexity using KL divergence between 3D point distributions in sliding windows.

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- LAS file support requires [laspy](https://laspy.readthedocs.io/)

### Quick Install
```bash
pip install numpy scipy scikit-learn laspy
```

## 🚀 Basic Usage

```bash
python VSRE.py --file test.las --ws 10.0 --zbox 20
```

## ⚙️ Full Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--file` | Path to input LAS file | **Required** |
| `--ws` | Window size (meters) | 10.0 |
| `--zbox` | Vertical divisions per window | 20 |
| `--ratio` | Step-to-window ratio (1=no overlap) | 1.0 |
| `--normalize_terrain` | Enable terrain normalization | False |
| `--grid_size` | Ground extraction grid size | 1.0 |
| `--use_gmm` | Use GMM for ground modeling | False |
| `--n_components` | GMM components (if use_gmm=True) | 1 |

## 📊 Example Output

```
Vegetation Structure Relative Entropy: 4.328571
```

## 📚 Methodology

1. **Data Loading**
   - Reads LAS files using laspy
   - Optional terrain removal (RANSAC/GMM)

2. **3D Analysis**
   - Sliding window segmentation
   - 3D histogram generation
   - KL divergence calculation

3. **VSRE Scoring**
   - Window-to-window comparisons
   - Averaged complexity metric

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

## ❓ Support

For issues, please open a GitHub ticket.
