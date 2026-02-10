# Census Income Classification and Clustering

This repository contains machine learning pipelines for Census dataset analysis, including XGBoost classification for income prediction and K-Means clustering for population segmentation.

## Project Structure

```
├── classification_xgboost.py    # XGBoost classification pipeline
├── kmeans_cluster.py            # K-Means clustering pipeline
├── environment.yml              # Conda environment specification
├── segmentation/                # Directory with segmentation helper functions
├── classification/              # Directory with classification helper functions
├── data_plots/                  # Generated plots and visualizations
├── cluster_output/              # Output of the Clusters generated via K-means
├── feature_importance.py        # File with Random Forest that calculates importance of each feature
├── plot_features.py             # File that plots the distribution of each feature
└── README.md
```
## Installation

Option 1

1. Enter the repository:
```bash
cd JPMC_Assign
```

2. Create the conda environment from the provided YAML file:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate JPMC_Assign
```

Option 2

1. Enter the repository:
```bash
cd JPMC_Assign
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```
## Usage

### Classification

Run the XGBoost classification pipeline for income prediction:

```bash
python classification_xgboost.py
```

### Clustering

Run the K-Means clustering pipeline for population segmentation:

```bash
python kmeans_cluster.py
```
