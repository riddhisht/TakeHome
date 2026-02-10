import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams

CSV_PATH = "census_bureau.csv"
OUTPUT_DIR = "data_plots"  
MAX_CATEGORIES = 25    

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
sns.set_style("ticks", {'font.family': 'serif'}) 
pub_palette = ["#2C3E50", "#E74C3C"]  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

# Preprocessing
df["target_raw"] = df["label"].map(lambda x: 1 if str(x).startswith("50000") else 0)
df["Class Label"] = df["target_raw"].map({0: "Class 0 (<50K)", 1: "Class 1 (>=50K)"})

raw_features = [c for c in df.columns if c not in ["label", "target_raw", "Class Label"]]
for col in raw_features:
    x = df[col]
    if x.isna().all():
        continue

    safe_name = col.replace(" ", "_").replace("-", "_").replace("/", "_").replace("?", "")
    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")

    # Numeric Features -Violin Plot
    if pd.api.types.is_numeric_dtype(x):
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))
        
        sns.violinplot(data=df,x="Class Label",y=col,hue="Class Label",palette=pub_palette,inner="quart",linewidth=1.2,legend=False,ax=ax)
        
        # Aesthetics
        ax.set_title(f"Distribution of {col} by Class", fontweight='bold', pad=15)
        ax.set_xlabel("") 
        ax.set_ylabel(col.replace("_", " ").title())
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        # Fill NaNs
        df[col] = df[col].fillna("Unknown")
        
        n_unique = df[col].nunique()
        if n_unique > MAX_CATEGORIES:
            continue

        order = df[col].value_counts().index
        fig_height = max(5, n_unique * 0.4)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        sns.countplot(data=df, y=col, hue="Class Label", palette=pub_palette, order=order,edgecolor="black", linewidth=0.5,ax=ax)

        # Aesthetics
        ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontweight='bold', pad=15)
        ax.set_ylabel("")
        ax.set_xlabel("Count")
        sns.move_legend(ax, "lower right")
        ax.xaxis.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True) 
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
