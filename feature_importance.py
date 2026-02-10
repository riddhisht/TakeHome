import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("census_bureau.csv")

nan_rules = {
    "Not in universe": df.columns.tolist(),
    0: ["wage per hour", "detailed occupation recode", "detailed industry recode"],
    "All other": ["hispanic origin"],
    "Nonfiler": ["tax filer stat"],
    "?": [
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "country of birth father",
        "country of birth mother",
    ],
    "Not in universe or children": ["major industry code"],
    "Not in universe under 1 year old": ["live in this house 1 year ago"],
    "Do not know": ["hispanic origin"]
}

df_nan = df.copy()
for v, cols in nan_rules.items():
    for c in cols:
        if c in df_nan.columns:
            df_nan[c] = df_nan[c].replace(v, np.nan)

X = df_nan.drop(columns=["label"])
y = df_nan["label"]

num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)

clf.fit(X_train, y_train)

feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
importances = clf.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"feature": feature_names,"importance": importances})
importance_df["orig_feature"] = (importance_df["feature"].str.replace(r"^(num|cat)__", "", regex=True).str.split("_").str[0])
feature_importance_agg = (importance_df.groupby("orig_feature")["importance"].sum().sort_values(ascending=False))

print(feature_importance_agg)
