
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

def custom_blue_green(values):
    cmap = LinearSegmentedColormap.from_list('blue_green', ['#a8d5ba', '#4a7c59'])
    return [cmap(v) for v in values]
def compute_profiles_from_processed(df_proc, labels, cat_cols, num_cols, top_k_cat, top_k_num, out_dir):

    df_proc = pd.DataFrame(df_proc)
    df_proc = df_proc.copy()
    df_proc.columns = [str(c) for c in df_proc.columns]  

    df_proc['_cluster_internal_temp'] = labels
    overall_props = df_proc.drop(columns=['_cluster_internal_temp']).mean(axis=0)
    cluster_means = df_proc.groupby('_cluster_internal_temp').mean()

    def matched_feature_cols(orig_cat):
        patterns = [f"{orig_cat}_", f"{orig_cat}__", f"{orig_cat}-", f"{orig_cat} "]
        matched = [c for c in df_proc.columns if any(c.startswith(p) for p in patterns)]
        return matched

    cluster_profiles = {}
    rows = []

    for cl in sorted(cluster_means.index):
        cm = cluster_means.loc[cl]
        # categorical lifts
        cat_lifts = []
        for orig_cat in (cat_cols or []):
            matched = matched_feature_cols(orig_cat)
            if not matched:
                continue
            for m in matched:
                if m not in cm.index or m not in overall_props.index:
                    continue
                cluster_prop = float(cm[m])
                overall_prop = float(overall_props[m])
                lift = (cluster_prop / overall_prop) if overall_prop > 0 else np.inf
                cat_lifts.append((orig_cat, m, cluster_prop, overall_prop, lift))
        cat_lifts_sorted = sorted(cat_lifts, key=lambda x: x[4], reverse=True)
        top_cat = cat_lifts_sorted[:top_k_cat]

        # numeric effects 
        num_effects = []
        for num in (num_cols or []):
            num_str = str(num)
            if num_str not in df_proc.columns:
                continue
            global_mean = df_proc[num_str].mean()
            global_std = df_proc[num_str].std(ddof=0) if df_proc[num_str].std(ddof=0) > 0 else 1.0
            cl_mean = cm[num_str]
            effect = (cl_mean - global_mean) / global_std
            num_effects.append((num, float(cl_mean), float(global_mean), float(effect)))
        num_effects_sorted = sorted(num_effects, key=lambda x: abs(x[3]), reverse=True)[:top_k_num]

        cluster_profiles[cl] = {
            'size': int((labels == cl).sum()),
            'top_categorical_lifts': top_cat,
            'top_numeric_effects': num_effects_sorted
        }

        for (orig_cat, m, cprop, oprop, lift) in top_cat:
            rows.append({
                'cluster': cl,
                'type': 'categorical',
                'feature_group': orig_cat,
                'encoded_col': m,
                'cluster_prop': cprop,
                'overall_prop': oprop,
                'lift': lift,
                'cluster_size': cluster_profiles[cl]['size']
            })
        for (num, clm, gmean, eff) in num_effects_sorted:
            rows.append({
                'cluster': cl,
                'type': 'numeric',
                'feature': num,
                'cluster_mean': clm,
                'global_mean': gmean,
                'effect_size': eff,
                'cluster_size': cluster_profiles[cl]['size']
            })

    profiles_df = pd.DataFrame(rows)
    df_proc_with_cluster = df_proc.drop(columns=['_cluster_internal_temp']).copy()
    df_proc_with_cluster['cluster'] = labels
    df_proc_with_cluster.to_csv(os.path.join(out_dir, 'processed_features_with_cluster.csv'), index=False)
    pd.DataFrame({'cluster': labels}).to_csv(os.path.join(out_dir, 'cluster_assignments.csv'), index=False)
    profiles_df.to_csv(os.path.join(out_dir, 'cluster_profiles.csv'), index=False)

    return profiles_df, cluster_profiles

def plot_pca(vis_df, out_dir):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=vis_df,x='pc1',y='pc2',        hue='cluster',palette='tab10',s=25,alpha=0.7)
    plt.title("PCA Cluster Visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_clusters.png"), dpi=150)
    plt.close()

def plot_cluster_profiles(profiles_df, clusters, out_dir, top_k=10):

    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("ticks")

    # Categorical Lifts
    cat_profiles = profiles_df[profiles_df['type'] == 'categorical']
    for cl in clusters:
        top = (
            cat_profiles[cat_profiles['cluster'] == cl]
            .sort_values('lift', ascending=False)
            .head(top_k)
        )
        if top.empty: continue

        fig, ax = plt.subplots(figsize=(8, 0.5 * len(top) + 2))
        bar_colors = custom_blue_green(np.linspace(0.8, 0.2, len(top)))

        sns.barplot(
            x='lift', y='encoded_col', data=top,
            ax=ax, hue='encoded_col', palette=bar_colors, edgecolor='black', linewidth=0.5, legend=False
        )
        ax.axvline(1.0, color='#E74C3C', linestyle='--', linewidth=1.2, alpha=0.8)
        
        ax.set_title(f'Cluster {cl}: Top Categorical Lifts', fontweight='bold', pad=15)
        ax.set_xlabel('Lift Ratio (Reference = 1.0)')
        ax.set_ylabel('')
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cluster_{cl}_categorical_lifts.png'), dpi=300)
        plt.close()

    # Numeric Effect Sizes
    num_profiles = profiles_df[profiles_df['type'] == 'numeric']
    for cl in clusters:
        topn = (
            num_profiles[num_profiles['cluster'] == cl]
            .sort_values('effect_size', key=abs, ascending=False)
            .head(top_k)
        )
        if topn.empty: continue

        fig, ax = plt.subplots(figsize=(8, 0.5 * len(topn) + 2))
        v_min, v_max = topn['effect_size'].min(), topn['effect_size'].max()
        norm = plt.Normalize(vmin=min(v_min, -0.1), vmax=max(v_max, 0.1))
        colors_mapped = custom_blue_green(norm(topn['effect_size'].values))

        sns.barplot(
            x='effect_size', y='feature', data=topn,
            ax=ax, palette=colors_mapped, edgecolor='black', linewidth=0.5
        )
        
        ax.axvline(0, color='black', linewidth=1)
        
        ax.set_title(f'Cluster {cl}: Numeric Feature Deviations', fontweight='bold', pad=15)
        ax.set_xlabel('Effect Size (Standard Deviations)')
        ax.set_ylabel('')
        
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cluster_{cl}_numeric_effects.png'), dpi=300)
        plt.close()
