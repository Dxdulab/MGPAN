"""Evaluation metrics and plotting helpers for MGPAN."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve


def find_best_threshold_with_recall_constraint(
    y_true,
    pred_probs,
    min_recall: float = 0.6,
):
    precision, recall, thresholds = precision_recall_curve(y_true, pred_probs)
    precision = precision[:-1]
    recall = recall[:-1]

    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    valid = recall >= min_recall
    best_idx = np.argmax(f1) if valid.sum() == 0 else np.argmax(f1 * valid)

    return thresholds[best_idx], f1[best_idx], precision[best_idx], recall[best_idx]


def plot_training_curves(train_loss, train_auc, save_dir="figures"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_loss, marker="o", color="tab:blue")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_dir / "train_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_auc, marker="o", color="tab:green")
    plt.title("Training AUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0, 1.0)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_dir / "train_auc_curve.png", dpi=300)
    plt.close()


def plot_test_evaluation(
    fpr,
    tpr,
    precision_curve,
    recall_curve,
    cm,
    test_auc,
    save_dir="figures",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "test_ROC_curve.png", dpi=300)
    plt.close()

    pr_auc = auc(recall_curve, precision_curve)
    plt.figure(figsize=(6, 6))
    plt.plot(
        recall_curve,
        precision_curve,
        color="green",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_dir / "test_PR_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_cross_validation_summary(
    fpr,
    tpr,
    tpr_std,
    precision_curve,
    prec_std,
    recall_curve,
    cm,
    mean_auc,
    std_auc,
    save_dir="figures",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tpr_upper = np.minimum(tpr + tpr_std, 1)
    tpr_lower = np.maximum(tpr - tpr_std, 0)

    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})",
    )
    plt.fill_between(fpr, tpr_lower, tpr_upper, color="darkorange", alpha=0.2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "mean_ROC_curve.png", dpi=300)
    plt.close()

    pr_auc = auc(recall_curve, precision_curve)
    prec_upper = np.minimum(precision_curve + prec_std, 1)
    prec_lower = np.maximum(precision_curve - prec_std, 0)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_curve, precision_curve, color="green", lw=2, label=f"Mean PR (AUC = {pr_auc:.3f})")
    plt.fill_between(recall_curve, prec_lower, prec_upper, color="green", alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mean Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_dir / "mean_PR_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Mean Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / "mean_confusion_matrix.png", dpi=300)
    plt.close()


def summarize_cv_curves(all_fpr, all_tpr, all_prec, all_recall, all_cm, all_auc):
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_recall = np.linspace(0, 1, 100)
    interp_precs = [
        np.interp(mean_recall, recall, precision)
        for recall, precision in zip(all_recall, all_prec)
    ]
    mean_prec = np.mean(interp_precs, axis=0)
    std_prec = np.std(interp_precs, axis=0)

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "std_tpr": std_tpr,
        "mean_recall": mean_recall,
        "mean_prec": mean_prec,
        "std_prec": std_prec,
        "mean_cm": np.mean(np.stack(all_cm), axis=0),
        "mean_auc": np.mean(all_auc),
        "std_auc": np.std(all_auc),
    }


def ensure_dir(path: Union[str, os.PathLike]) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)
