# Census Income Classification and Clustering

This repository contains machine learning pipelines for Census dataset analysis, including XGBoost classification for income prediction and K-Means clustering for population segmentation.

---

## Project structure

```
├── classification_xgboost.py      # Train + save XGBoost model
├── evaluate_classification.py     # Load saved model, run evaluation on a CSV (CLI)
├── kmeans_cluster.py              # K-Means clustering pipeline
├── environment.yml                # Conda environment specification
├── classification/                # Preprocessing & helper functions
├── segmentation/                  # Clustering helper functions
├── data_plots/                    # Generated plots and visualizations
├── cluster_output/                # K-Means output files
├── feature_importance.py          # Random Forest feature importance
├── plot_features.py               # Feature distribution plots
└── README.md
```

---

## Quick start

### Option A — Conda

```bash
cd TakeHome
conda env create -f environment.yml
conda activate TakeHome
```

### Option B — venv + pip

```bash
cd TakeHome
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

---

## Data

Provide a CSV with Census-style features and an income label. Example filename used in the repo:

```
census_bureau.csv
```

Preprocessing is centralized in `classification/preprocess.py` and reused by training and evaluation.

---

## Classification (XGBoost)

### Train

Runs training with early stopping and saves the model.

```bash
python classification_xgboost.py --csv path/to/census_bureau.csv
```

### Evaluate

Load a saved model and evaluate a CSV. Threshold is configurable depending on if you want precision first or recall first strategy. (default = 0.8).

```bash
python evaluate_classification.py --csv path/to/new_data.csv
python evaluate_classification.py --csv path/to/new_data.csv --threshold 0.9
```

---

## Clustering (K-Means)

Run K-Means to generate cluster assignments and visualizations:

```bash
python kmeans_cluster.py --csv path/to/census_bureau.csv
```

---
