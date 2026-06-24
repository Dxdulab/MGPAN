"""Plotting utilities refactored from the original utils.py."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc
def plot_training_curves(train_loss, train_auc, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_loss)+1)

    plt.figure(figsize=(8,5))
    sns.lineplot(x=epochs, y=train_loss, marker='o', color='tab:blue')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_loss_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8,5))
    sns.lineplot(x=epochs, y=train_auc, marker='o', color='tab:green')
    plt.title("Training AUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0, 1.0)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_auc_curve.png", dpi=300)
    plt.show()


def plot_test_evaluation(fpr, tpr, precision_curve, recall_curve, cm, test_auc, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/test_ROC_curve.png", dpi=300)
    plt.show()

    pr_auc = auc(recall_curve, precision_curve)
    plt.figure(figsize=(6,6))
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/test_PR_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)
    plt.show()

def plot_testcv_evaluation(
        fpr, 
        tpr, 
        tpr_std, 
        precision_curve, 
        prec_std, 
        recall_curve, 
        cm, 
        mean_auc, std_auc,   # ✅ 新增参数
        save_dir='figures'
    ):
    os.makedirs(save_dir, exist_ok=True)

    tpr_upper = np.minimum(tpr + tpr_std, 1)
    tpr_lower = np.maximum(tpr - tpr_std, 0)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')  # ✅ 显示 ±std

    plt.fill_between(fpr, tpr_lower, tpr_upper,
                     color='darkorange', alpha=0.2,
                     label='±1 std. dev.')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve (5-Fold)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mean_ROC_curve.png", dpi=300)
    plt.show()

    pr_auc = auc(recall_curve, precision_curve)
    prec_upper = np.minimum(precision_curve + prec_std, 1)
    prec_lower = np.maximum(precision_curve - prec_std, 0)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_curve, precision_curve, color='green', lw=2,
             label=f'Mean PR (AUC = {pr_auc:.3f})')
    plt.fill_between(recall_curve, prec_lower, prec_upper,
                     color='green', alpha=0.2, label='±1 std. dev.')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Mean Precision-Recall Curve (5-Fold)')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mean_PR_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Mean Confusion Matrix (5-Fold)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mean_confusion_matrix.png", dpi=300)
    plt.show()

