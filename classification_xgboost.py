import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, log_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classification.preprocess import preprocess
import joblib
  
# Load data
df = pd.read_csv("census_bureau.csv")

# Preprocess data
X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test = preprocess(df)

# ensure 1d numpy arrays
y_train_arr = np.asarray(y_train).ravel()
y_val_arr = np.asarray(y_val).ravel()
y_test_arr = np.asarray(y_test).ravel()

# use XGBoost DMatrix
dtrain = xgb.DMatrix(X_train_enc, label=y_train_arr)
dval = xgb.DMatrix(X_val_enc, label=y_val_arr)
dtest = xgb.DMatrix(X_test_enc, label=y_test_arr)

# Compute scale_pos_weight from training set 
n_pos = y_train_arr.sum()
n_neg = len(y_train_arr) - n_pos
scale_pos_weight = float(n_neg / max(1.0, n_pos))


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.05,           
    "max_depth": 6,
    "min_child_weight": 1,
    "scale_pos_weight": 15,
    "seed": 42,
}

# train with early stopping
evals_result = {}
watchlist = [(dtrain, "train"), (dval, "eval")]
num_boost_round = 2000
early_stopping_rounds = 50
bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=25, evals_result=evals_result)

# Save model
bst.save_model("xgb_model.json")
joblib.dump(bst, "xgb_model.joblib")
# Predict probs on validation/test
val_probs = bst.predict(dval)
test_probs = bst.predict(dtest)

#Tunable threshold
best_thresh = 0.95

# Evaluate on test set using that threshold
test_preds = (test_probs > best_thresh).astype(int)

logloss_test = log_loss(y_test_arr, test_probs)
auc_test = roc_auc_score(y_test_arr, test_probs)
acc_test = accuracy_score(y_test_arr, test_preds)
prec0 = precision_score(y_test_arr, test_preds, pos_label=0)
prec1 = precision_score(y_test_arr, test_preds, pos_label=1)
rec0 = recall_score(y_test_arr, test_preds, pos_label=0)
rec1 = recall_score(y_test_arr, test_preds, pos_label=1)

#print the results
print(f"Test Log Loss: {logloss_test}")
print(f"Test AUC: {auc_test}")
print(f"Test Accuracy (thresh = {best_thresh}): {acc_test}")
print(f"Precision class 0: {prec0}, class 1: {prec1}")
print(f"Recall class 0: {rec0}, class 1: {rec1}")

# Plot Learning Curve (Log Loss)
plt.figure(figsize=(10, 6))
plt.plot(evals_result['train']['logloss'], label='Training Loss')
plt.plot(evals_result['eval']['logloss'], label='Validation Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('Learning Curve (Log Loss)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test_arr, test_probs)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC Curve (AUC = {auc_test:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Performance')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot Predicted Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(test_probs[y_test_arr == 0], bins=50, alpha=0.7, label='Class 0 (<$50K)', color='gray', density=True)
plt.hist(test_probs[y_test_arr == 1], bins=50, alpha=0.7, label='Class 1 (>=$50K)', color='red', density=True)
plt.xlabel('Predicted Probability of Class 1')
plt.ylabel('Percent of Samples (%)')
plt.title('XGBoost Predicted Probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
plt.show()