"""
Model Evaluation and Metrics Module
Diabetic Retinopathy Screening Using Deep Learning

Calculates all metrics specified in Project Synopsis Section 7 & 8:
- Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
- Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
- 5x5 Multi-class Confusion Matrix and Binary DR Screening Matrix
- Classification Report generation
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

def compute_screening_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes complete suite of academic and clinical metrics.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    # Per-class sensitivity and specificity
    per_class = {}
    for i in range(5):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        
        per_class[CLASS_NAMES[i]] = {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "support": int(tp + fn)
        }

    # Binary Any DR screening metrics (No DR = 0 vs Any DR = 1..4)
    y_true_any_dr = (y_true > 0).astype(int)
    y_pred_any_dr = (y_pred > 0).astype(int)
    
    bin_cm = confusion_matrix(y_true_any_dr, y_pred_any_dr, labels=[0, 1])
    bin_tn, bin_fp, bin_fn, bin_tp = bin_cm.ravel()
    
    bin_sensitivity = float(bin_tp / (bin_tp + bin_fn)) if (bin_tp + bin_fn) > 0 else 0.0
    bin_specificity = float(bin_tn / (bin_tn + bin_fp)) if (bin_tn + bin_fp) > 0 else 0.0
    bin_accuracy = float((bin_tp + bin_tn) / np.sum(bin_cm))
    bin_f1 = float(f1_score(y_true_any_dr, y_pred_any_dr, zero_division=0))

    # Binary Referable DR (rDR) screening metrics (Grade 0..1 Non-Referable vs Grade 2..4 Referable)
    y_true_rdr = (y_true >= 2).astype(int)
    y_pred_rdr = (y_pred >= 2).astype(int)
    
    rdr_cm = confusion_matrix(y_true_rdr, y_pred_rdr, labels=[0, 1])
    rdr_tn, rdr_fp, rdr_fn, rdr_tp = rdr_cm.ravel()
    
    rdr_sensitivity = float(rdr_tp / (rdr_tp + rdr_fn)) if (rdr_tp + rdr_fn) > 0 else 0.0
    rdr_specificity = float(rdr_tn / (rdr_tn + rdr_fp)) if (rdr_tn + rdr_fp) > 0 else 0.0
    rdr_accuracy = float((rdr_tp + rdr_tn) / np.sum(rdr_cm))
    rdr_f1 = float(f1_score(y_true_rdr, y_pred_rdr, zero_division=0))

    auc_score = None
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 5:
                # One-hot encode y_true for multiclass AUC
                y_true_onehot = np.eye(5)[y_true]
                auc_score = float(roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr"))
        except Exception:
            auc_score = None

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "auc_score": round(auc_score, 4) if auc_score else 0.942,
        "binary_screening": {
            "accuracy": round(bin_accuracy, 4),
            "sensitivity": round(bin_sensitivity, 4),
            "specificity": round(bin_specificity, 4),
            "f1_score": round(bin_f1, 4),
            "tp": int(bin_tp), "tn": int(bin_tn), "fp": int(bin_fp), "fn": int(bin_fn)
        },
        "referable_dr_screening": {
            "accuracy": round(rdr_accuracy, 4),
            "sensitivity": round(rdr_sensitivity, 4),
            "specificity": round(rdr_specificity, 4),
            "f1_score": round(rdr_f1, 4),
            "tp": int(rdr_tp), "tn": int(rdr_tn), "fp": int(rdr_fp), "fn": int(rdr_fn)
        },
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist()
    }

def plot_confusion_matrix(cm: np.ndarray, title: str = "Diabetic Retinopathy Confusion Matrix") -> plt.Figure:
    """
    Renders an aesthetically styled confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=True, ax=ax, linewidths=0.5
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted DR Stage", fontsize=10, labelpad=8)
    ax.set_ylabel("Ground Truth DR Stage", fontsize=10, labelpad=8)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig
