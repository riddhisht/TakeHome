import os
import pandas as pd
from segmentation.preprocess import apply_mappings, build_preprocessor
from segmentation.analysis import compute_profiles_from_processed, plot_pca, plot_cluster_profiles
from segmentation.model import run_clustering
import argparse

# Parse
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to .data file")
parser.add_argument("--columns", required=True, help="Path to .columns file")
args = parser.parse_args()

OUT_DIR = "cluster_output"

N_CLUSTERS = 8
RANDOM_STATE = 42

TOP_K_CAT = 10
TOP_K_NUM = 10
os.makedirs(OUT_DIR, exist_ok=True)

# Read column names from .columns file
with open(args.columns, 'r') as f:
    column_names = [line.strip() for line in f if line.strip()]

# Read data from .data file
df = pd.read_csv(args.data, names=column_names, header=None)

# --- load & map ---
df = apply_mappings(df)
label_col = 'positive' if 'positive' in df.columns else None
df_clust = df.drop(columns=[label_col]) if label_col else df

# build preprocessor
ct, cat_cols, num_cols = build_preprocessor(df_clust)
X_proc = ct.fit_transform(df_clust)
ohe = ct.named_transformers_['cat']
ohe_names = ohe.get_feature_names_out(cat_cols).tolist() if hasattr(ohe, 'get_feature_names_out') else []
feature_names = ohe_names + num_cols
X_proc_df = pd.DataFrame(X_proc, columns=[str(c) for c in feature_names], index=df_clust.index)

# cluster
labels, X_pca, km, pca = run_clustering(X_proc_df.values, N_CLUSTERS, RANDOM_STATE)

# attach cluster label 
df['cluster'] = labels
df.to_csv(os.path.join(OUT_DIR, "original_with_cluster.csv"), index=False)

# profiling
profiles_df, cluster_profiles_dict = compute_profiles_from_processed(
    df_proc=X_proc_df,
    labels=labels,
    cat_cols=cat_cols,
    num_cols=num_cols,
    top_k_cat=TOP_K_CAT,
    top_k_num=TOP_K_NUM,
    out_dir=OUT_DIR
)

# visualization
vis_df = pd.DataFrame({
    'pc1': X_pca[:, 0],
    'pc2': X_pca[:, 1],
    'cluster': labels
})
plot_pca(vis_df, OUT_DIR)
unique_clusters = sorted(pd.unique(labels))
plot_cluster_profiles(
    profiles_df=profiles_df,
    clusters=unique_clusters,
    out_dir=OUT_DIR,
    top_k=10
)
print("Done. Outputs in", OUT_DIR)
