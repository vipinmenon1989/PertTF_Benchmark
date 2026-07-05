import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import scanpy as sc
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, f1_score, precision_score, 
    recall_score, accuracy_score, roc_auc_score, 
    precision_recall_curve, auc, roc_curve
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

# Suppress Optuna verbose optimization tracking output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# 1. SETUP MASTER TRACKING OUTPUT DIRECTORY
# ==============================================================================
CELLTYPE_DIR = "./celltype_split"
os.makedirs(CELLTYPE_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA INGESTION & STANDARD SCANPY PREPROCESSING PIPELINE
# ==============================================================================
print("Loading raw AnnData object...")
adata = sc.read_h5ad("object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad")

print("Executing standard single-cell preprocessing pipeline...")
if "counts" in adata.layers:
    adata.X = adata.layers["counts"].copy()

# Step A: Target Sum Depth Normalization
sc.pp.normalize_total(adata, target_sum=1e4)

# Step B: Logarithmic Transformation
sc.pp.log1p(adata)

# Step C: Isolate 5000 Highly Variable Genes (HVGs) with seed control
print("Extracting top 5000 highly variable gene transcripts...")
sc.pp.highly_variable_genes(
    adata, 
    n_top_genes=5000, 
    flavor="seurat", 
    subset=True
)

# Step D: Dimensionality Reduction via 100 Principal Components (PCs) with seed control
print("Computing top 100 Principal Components...")
sc.pp.pca(adata, n_comps=100, random_state=42)

# Extract Features (100 Dimensional Matrix)
X = adata.obsm["X_pca"]
features = [f"PC_{i+1}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=features)

# Target vector encoding using celltype_2 column
le_cell = LabelEncoder()
y_cell = le_cell.fit_transform(adata.obs["celltype_2"])

n_classes = len(le_cell.classes_)
obj = "binary" if n_classes == 2 else "multiclass"
metric = "binary_logloss" if n_classes == 2 else "multi_logloss"


# ==============================================================================
# EVALUATION METRICS & PLOTTER UTILITIES
# ==============================================================================
def calculate_and_save_metrics(y_true, y_pred, y_prob, output_dir, prefix=""):
    if n_classes == 2:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        p, r, _ = precision_recall_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        aupr = auc(r, p)
    else:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        roc_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        
        aupr_list = []
        for i in range(n_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                aupr_list.append(auc(r, p))
        aupr = np.mean(aupr_list) if aupr_list else 0.0

    metrics_dict = {
        "Accuracy": acc, "Precision_Macro": prec, "Recall_Macro": rec,
        "F1_Score_Macro": f1, "ROC_AUC_Macro": roc_auc, "AUPR_Macro": aupr
    }
    filename = f"{prefix}evaluation_metrics.csv" if prefix else "evaluation_metrics.csv"
    pd.DataFrame([metrics_dict]).to_csv(os.path.join(output_dir, filename), index=False)


def generate_curves(y_true, y_prob, output_dir, prefix=""):
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes)) if n_classes > 2 else y_true
    colors = plt.cm.turbo(np.linspace(0, 1, n_classes))

    # --- 1. ROC Curve Generation ---
    fig, ax = plt.subplots(figsize=(10, 8))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc(fpr, tpr):.3f})", color="darkorange", lw=2)
    else:
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        valid_classes = 0
        for i in range(n_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, color=colors[i], alpha=0.65, lw=1.0, label=le_cell.classes_[i])
                mean_tpr += np.interp(fpr_grid, fpr, tpr)
                valid_classes += 1
        if valid_classes > 0:
            mean_tpr /= valid_classes
            macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            ax.plot(fpr_grid, mean_tpr, label=f"Macro-average (AUC = {macro_auc:.3f})", color="black", lw=3.0, linestyle="--")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{prefix.upper()} Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight='bold')
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=2, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}roc_curve.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{prefix}roc_curve.pdf"), format="pdf")
    plt.close()

    # --- 2. Precision-Recall (PR) Curve Generation ---
    fig, ax = plt.subplots(figsize=(10, 8))
    if n_classes == 2:
        p, r, _ = precision_recall_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        ax.plot(r, p, label=f"PR Curve (AUPR = {auc(r, p):.3f})", color="forestgreen", lw=2)
    else:
        aupr_list = []
        recall_grid = np.linspace(0.0, 1.0, 1000)
        mean_precision = np.zeros_like(recall_grid)
        valid_classes = 0
        for i in range(n_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                aupr_list.append(auc(r, p))
                ax.plot(r, p, color=colors[i], alpha=0.65, lw=1.0, label=le_cell.classes_[i])
                mean_precision += np.interp(recall_grid, r[::-1], p[::-1])
                valid_classes += 1
        if valid_classes > 0:
            mean_precision /= valid_classes
            ax.plot(recall_grid, mean_precision, label=f"Macro-average (AUPR = {np.mean(aupr_list):.3f})", color="black", lw=3.0, linestyle="--")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"{prefix.upper()} Precision-Recall (AUPR) Curve", fontsize=13, fontweight='bold')
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=2, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}pr_curve.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{prefix}pr_curve.pdf"), format="pdf")
    plt.close()


def optimize_hyperparameters(X_t_train, y_t_train, X_t_val, y_t_val, n_trials=50):
    def objective(trial):
        params = {
            "objective": obj, "metric": metric, "num_class": n_classes if n_classes > 2 else 1,
            "boosting_type": "gbdt", "n_jobs": -1, "verbose": -1, "random_state": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        dtrain = lgb.Dataset(X_t_train, label=y_t_train)
        dval = lgb.Dataset(X_t_val, label=y_t_val, reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=150, valid_sets=[dval], callbacks=[lgb.early_stopping(10, verbose=False)])
        preds = model.predict(X_t_val)
        y_pred = np.argmax(preds, axis=1) if n_classes > 2 else (preds > 0.5).astype(int)
        return f1_score(y_t_val, y_pred, average="macro")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# ==============================================================================
# PIPELINE: 5-FOLD STRATIFIED RANDOM CV SPLIT WITH GLOBAL OUT-OF-FOLD STORAGE
# ==============================================================================
print("\n=== EXECUTING PIPELINE: 5-FOLD STRATIFIED RANDOM CV FOR CELL TYPE ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Global tracking structures for out-of-fold pooled metrics
oof_preds = np.zeros(len(X_df), dtype=int)
if n_classes > 2:
    oof_probs = np.zeros((len(X_df), n_classes))
else:
    oof_probs = np.zeros((len(X_df), 2))

for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y_cell)):
    print(f"\n--- Processing Cell Type CV Fold {fold+1}/5 ---")
    X_train_rand, Y_train_rand = X_df.iloc[train_idx], y_cell[train_idx]
    X_test_rand, Y_test_rand = X_df.iloc[test_idx], y_cell[test_idx]

    TUNE_SIZE_RAND = min(15000, len(X_train_rand))
    _, X_tune_rand, _, y_tune_rand = train_test_split(X_train_rand, Y_train_rand, test_size=TUNE_SIZE_RAND/len(X_train_rand), stratify=Y_train_rand, random_state=42)
    X_r_train, X_r_val, y_r_train, y_r_val = train_test_split(X_tune_rand, y_tune_rand, test_size=0.2, stratify=y_tune_rand, random_state=42)

    best_params_rand = optimize_hyperparameters(X_r_train, y_r_train, X_r_val, y_r_val, n_trials=50)

    rand_train_dataset = lgb.Dataset(X_train_rand, label=Y_train_rand)
    final_params_rand = {"objective": obj, "metric": metric, "num_class": n_classes if n_classes > 2 else 1, "boosting_type": "gbdt", "n_jobs": -1, "verbose": -1, "random_state": 42, **best_params_rand}
    
    final_model_rand = lgb.train(final_params_rand, rand_train_dataset, num_boost_round=400)

    y_prob_rand = final_model_rand.predict(X_test_rand)
    
    if n_classes == 2 and y_prob_rand.ndim == 1:
        # Standardize 1D binary probabilities into standard 2D shape [P(c0), P(c1)]
        y_prob_rand_2d = np.zeros((len(y_prob_rand), 2))
        y_prob_rand_2d[:, 1] = y_prob_rand
        y_prob_rand_2d[:, 0] = 1.0 - y_prob_rand
        y_prob_rand = y_prob_rand_2d

    y_pred_rand = np.argmax(y_prob_rand, axis=1)

    # Store predictions into the global Out-of-Fold tracking indices
    oof_preds[test_idx] = y_pred_rand
    oof_probs[test_idx] = y_prob_rand

    fold_dir = os.path.join(CELLTYPE_DIR, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    with open(os.path.join(fold_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(Y_test_rand, y_pred_rand, labels=np.arange(n_classes), target_names=le_cell.classes_, zero_division=0))

    np.save(os.path.join(fold_dir, "y_true.npy"), Y_test_rand)
    np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred_rand)
    np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob_rand)
    pd.DataFrame(best_params_rand, index=[0]).to_csv(os.path.join(fold_dir, "best_params.csv"), index=False)

    calculate_and_save_metrics(Y_test_rand, y_pred_rand, y_prob_rand, fold_dir)
    generate_curves(Y_test_rand, y_prob_rand, fold_dir)

# ==============================================================================
# 4. COMPUTE TRUE OVERALL CROSS-VALIDATION METRICS (OUT-OF-FOLD)
# ==============================================================================
print("\n=== GENERATING MASTER OVERALL CROSS-VALIDATION PERFORMANCE REPORTS ===")

with open(os.path.join(CELLTYPE_DIR, "overall_classification_report.txt"), "w") as f:
    f.write(classification_report(y_cell, oof_preds, labels=np.arange(n_classes), target_names=le_cell.classes_, zero_division=0))

np.save(os.path.join(CELLTYPE_DIR, "oof_true.npy"), y_cell)
np.save(os.path.join(CELLTYPE_DIR, "oof_pred.npy"), oof_preds)
np.save(os.path.join(CELLTYPE_DIR, "oof_prob.npy"), oof_probs)

# Calculate overall macro metrics across all aggregated predictions
calculate_and_save_metrics(y_cell, oof_preds, oof_probs, CELLTYPE_DIR, prefix="overall_")
generate_curves(y_cell, oof_probs, CELLTYPE_DIR, prefix="overall_")

print(f"\nAll tasks finalized. Master cross-validation logs and curves compiled directly under: {CELLTYPE_DIR}")
