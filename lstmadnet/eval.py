import os
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score, confusion_matrix)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_model(model,
                   dataset,
                   evals_dir=None,
                   wandb_info=None,
                   dataset_path="",
                   checkpoint_path="",
                   verbose=True,
                   save=False):
    """
    Evaluates a trained model on a given dataset and optionally saves the evaluation report.

    Metrics computed:
        - Accuracy
        - Balanced accuracy
        - Recall (weighted)
        - Precision (weighted)
        - F1 score (weighted)
        - ROC AUC (weighted, one-vs-one)
        - Confusion matrix (normalized)

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataset (Dataset): The dataset for evaluation.
        evals_dir (str, optional): Directory for saving evaluation reports.
        wandb_info (dict, optional): W&B project info with keys "project_name" and "run_name".
        dataset_path (str): Path of the dataset being evaluated (for reporting).
        checkpoint_path (str): Path of the checkpoint used (for reporting).
        verbose (bool): Whether to display tensor shapes. Set False for LF/DF models.
        save (bool): If True, saves results to a JSON file. Defaults to False.

    Returns:
        dict: Evaluation metrics including 'acc', 'acc_b', 'rec', 'pre', 'f1', 'roc', 'cm'.
    """
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    for data, targets in dataloader:
        with torch.inference_mode():
            logits = model(data)
            targets_hat = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            print("Evaluating ...")
            if verbose:
                print(f"  data.shape: {data.shape}")
                print(f"  targets.shape: {targets.shape}")
                print(f"  logits.shape: {logits.shape}")

            report = {}
            report["checkpoint_path"] = checkpoint_path
            report["dataset_path"] = dataset_path
            report["acc"] = accuracy_score(y_pred=targets_hat, y_true=targets)
            report["acc_b"] = balanced_accuracy_score(y_pred=targets_hat, y_true=targets)
            report["rec"] = recall_score(y_pred=targets_hat, y_true=targets, average="weighted", zero_division=np.nan)
            report["pre"] = precision_score(y_pred=targets_hat, y_true=targets, average="weighted", zero_division=np.nan)
            report["f1"] = f1_score(y_pred=targets_hat, y_true=targets, average="weighted", zero_division=np.nan)
            report["roc"] = roc_auc_score(y_score=probs, y_true=targets, average="weighted", multi_class="ovo")
            report["cm"] = confusion_matrix(y_pred=targets_hat, y_true=targets, normalize="true")

            print(f"  Accuracy: {report['acc']:.4f}")
            print(f"  Balanced Accuracy: {report['acc_b']:.4f}")
            print(f"  F1 (weighted): {report['f1']:.4f}")
            print(f"  ROC AUC (weighted): {report['roc']:.4f}")

            if save:
                if evals_dir is None or wandb_info is None:
                    raise ValueError("evals_dir and wandb_info are required when save=True")
                
                target_dir = os.path.join(evals_dir, wandb_info["project_name"])
                if not Path(target_dir).is_dir():
                    Path(target_dir).mkdir(parents=True)
                    
                target_path = os.path.join(target_dir, f"report_{wandb_info['run_name']}.json")
                print(f"\n  Saving evaluation results to {target_path}")
                with open(target_path, "w") as f:
                    json.dump(obj=report, fp=f, cls=NumpyEncoder, indent=2)

    return report
