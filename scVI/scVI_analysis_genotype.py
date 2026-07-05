import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, f1_score, precision_score, 
    recall_score, accuracy_score, roc_auc_score, 
    precision_recall_curve, auc, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

# Suppress verbose scvi training logs to keep stdout clean
scvi.settings.verbosity = 0

# ==============================================================================
# 1. SETUP MASTER TRACKING OUTPUT DIRECTORIES
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
H5AD_PATH = PROJECT_ROOT / "object_integrated_assay3_annotated_final.modified.cleaned.updated.celltype2_only.h5ad"
SCANVI_DIR = Path(__file__).resolve().parent / "scanvi_genotype_split"
os.makedirs(SCANVI_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA INGESTION & ANCHORED SCANPY PREPROCESSING PIPELINE
# ==============================================================================
print("Loading raw AnnData object...")
adata = sc.read_h5ad(H5AD_PATH)

# Check and secure raw integer count state inside the anndata layer slot
if "counts" not in adata.layers:
    print("Backing up raw count matrix from .X slot into layers...")
    adata.layers["counts"] = adata.X.copy()

print("Running standard single-cell variable selection...")
# Perform normalization and log transform on .X for highly variable gene calculation
if "counts" in adata.layers:
    adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Subset the global anndata object down to 5000 genes
# This shifts and slices both .X and layers['counts'] automatically without index mismatches
sc.pp.highly_variable_genes(
    adata, 
    n_top_genes=5000, 
    flavor="seurat", 
    subset=True
)

# Enforce pure non-negative rounded integers within .X to comply with Negative Binomial requirements
if hasattr(adata.layers["counts"], "toarray"):
    adata.layers["counts"].data = np.round(adata.layers["counts"].data)
else:
    adata.layers["counts"] = np.round(adata.layers["counts"])

adata.X = adata.layers["counts"].copy()

# Target vector string and index encoding using genotype column
le_geo = LabelEncoder()
y_geo = le_geo.fit_transform(adata.obs["genotype"])

n_classes = len(le_geo.classes_)

# ==============================================================================
# EVALUATION METRICS & PLOTTER UTILITIES
# ==============================================================================
def calculate_and_save_metrics(y_true, y_pred, y_prob, output_dir, prefix=""):
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
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    colors = plt.cm.turbo(np.linspace(0, 1, n_classes))

    # --- 1. ROC Curve ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    valid_classes = 0
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=colors[i], alpha=0.65, lw=1.0, label=le_geo.classes_[i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)
            valid_classes += 1
    if valid_classes > 0:
        mean_tpr /= valid_classes
        macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        ax.plot(fpr_grid, mean_tpr, label=f"Macro-average (AUC = {macro_auc:.3f})", color="black", lw=3.0, linestyle="--")

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{prefix.upper()} ROC Curve [scANVI Genotype Baseline]", fontsize=13, fontweight='bold')
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}roc_curve.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{prefix}roc_curve.pdf"), format="pdf")
    plt.close()

    # --- 2. Precision-Recall Curve ---
    fig, ax = plt.subplots(figsize=(10, 8))
    aupr_list = []
    recall_grid = np.linspace(0.0, 1.0, 1000)
    mean_precision = np.zeros_like(recall_grid)
    valid_classes = 0
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            p, r, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            aupr_list.append(auc(r, p))
            ax.plot(r, p, color=colors[i], alpha=0.65, lw=1.0, label=le_geo.classes_[i])
            mean_precision += np.interp(recall_grid, r[::-1], p[::-1])
            valid_classes += 1
    if valid_classes > 0:
        mean_precision /= valid_classes
        ax.plot(recall_grid, mean_precision, label=f"Macro-average (AUPR = {np.mean(aupr_list):.3f})", color="black", lw=3.0, linestyle="--")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"{prefix.upper()} Precision-Recall (AUPR) Curve [scANVI Genotype Baseline]", fontsize=13, fontweight='bold')
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}pr_curve.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{prefix}pr_curve.pdf"), format="pdf")
    plt.close()


# ==============================================================================
# 3. OUTER PIPELINE LOOP: 5-FOLD STRATIFIED RANDOM SPLIT WITH GLOBAL STORAGE
# ==============================================================================
print("\n=== EXECUTING STANDALONE DEEP LEARNING CV PIPELINE ON GPU ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Global tracking matrix initialization for out-of-fold pooled metrics
oof_preds = np.zeros(len(adata), dtype=int)
oof_probs = np.zeros((len(adata), n_classes))

for fold, (train_idx, test_idx) in enumerate(skf.split(adata.X, y_geo)):
    print(f"\n--- Processing scANVI Genotype Classifier Fold {fold+1}/5 ---")
    
    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()
    
    # Step A: Setup and Train underlying unsupervised scVI model
    scvi.model.SCVI.setup_anndata(adata_train)
    scvi_seed_model = scvi.model.SCVI(adata_train, n_latent=100)
    scvi_seed_model.train(
        max_epochs=150, 
        early_stopping=True, 
        accelerator="gpu", 
        enable_progress_bar=False
    )
    
    # Step B: Spin up semi-supervised classifier (scANVI) from seed weights
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_seed_model,
        labels_key="genotype",
        unlabeled_category="Unknown"
    )
    scanvi_model.train(
        max_epochs=30, 
        accelerator="gpu", 
        enable_progress_bar=False
    )
    
    # Step C: Evaluate natively on the withheld fold partition
    y_pred_str = scanvi_model.predict(adata_test)
    y_prob_unseen = scanvi_model.predict(adata_test, soft=True) 
    
    # Map predictions back to uniform integer IDs for global tracking arrays
    y_true_fold = y_geo[test_idx]
    y_pred_fold = le_geo.transform(y_pred_str)
    
    # Store fold results into the out-of-fold master matrix slots
    oof_preds[test_idx] = y_pred_fold
    oof_probs[test_idx] = y_prob_unseen.values
    
    # Save isolated fold diagnostics
    fold_dir = os.path.join(SCANVI_DIR, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    
    with open(os.path.join(fold_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true_fold, y_pred_fold, labels=np.arange(n_classes), target_names=le_geo.classes_, zero_division=0))
        
    np.save(os.path.join(fold_dir, "y_true.npy"), y_true_fold)
    np.save(os.path.join(fold_dir, "y_pred.npy"), y_pred_fold)
    np.save(os.path.join(fold_dir, "y_prob.npy"), y_prob_unseen.values)
    
    calculate_and_save_metrics(y_true_fold, y_pred_fold, y_prob_unseen.values, fold_dir)
    generate_curves(y_true_fold, y_prob_unseen.values, fold_dir)
    print(f"Fold {fold+1} localized logs finalized.")

# ==============================================================================
# 4. COMPUTE TRUE OVERALL CROSS-VALIDATION METRICS (OUT-OF-FOLD POOLED SUMMARY)
# ==============================================================================
print("\n=== GENERATING MASTER OVERALL CROSS-VALIDATION PERFORMANCE REPORTS ===")

with open(os.path.join(SCANVI_DIR, "overall_classification_report.txt"), "w") as f:
    f.write(classification_report(y_geo, oof_preds, labels=np.arange(n_classes), target_names=le_geo.classes_, zero_division=0))

np.save(os.path.join(SCANVI_DIR, "oof_true.npy"), y_geo)
np.save(os.path.join(SCANVI_DIR, "oof_pred.npy"), oof_preds)
np.save(os.path.join(SCANVI_DIR, "oof_prob.npy"), oof_probs)

# Execute aggregated performance metric extraction across all pooled folds
calculate_and_save_metrics(y_geo, oof_preds, oof_probs, SCANVI_DIR, prefix="overall_")
generate_curves(y_geo, oof_probs, SCANVI_DIR, prefix="overall_")

print(f"\nAll tasks finalized. Master OOF cross-validation logs and curves compiled directly under: {SCANVI_DIR}")
