import os
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch.nn.utils import clip_grad_norm_


def _train_epoch(model,
                 dataloader_tr,
                 device,
                 move_data_to_device,
                 optim_fcn,
                 loss_fcn,
                 acc_fcn,
                 clip_grad,
                 acc_fcn_type="torcheval"):
    """
    Trains the model for one epoch on the training dataset.
    
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        dataloader_tr (torch.utils.data.DataLoader): Dataloader for the training set.
        device (str or torch.device): Device on which to run the training.
        move_data_to_device (bool): Whether to move data and targets to device.
            Should be False when data is received as dict objects (LF & DF).
        optim_fcn (torch.optim.Optimizer): Optimizer for updating weights.
        loss_fcn (callable): Loss function for backpropagation.
        acc_fcn (callable): Accuracy function.
        clip_grad (bool): Whether to apply gradient clipping.
        acc_fcn_type (str): Accuracy function type ("torcheval" or "sklearn").
    
    Returns:
        tuple: (accumulated_loss, accumulated_accuracy) for the epoch.
    """
    model = model.to(device=device)
    model.train()
    _loss, _acc = 0, 0
    for data_batch, targets_batch in dataloader_tr:
        if move_data_to_device:
            data_batch = data_batch.to(device=device)
            targets_batch = targets_batch.to(device=device)

        optim_fcn.zero_grad()
        logits_batch = model(data_batch)

        loss_batch = loss_fcn(input=logits_batch, target=targets_batch)
        loss_batch.backward()
        if clip_grad:
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optim_fcn.step()

        _loss += loss_batch.detach().item()

        if acc_fcn_type == "torcheval":
            _acc += acc_fcn(input=logits_batch, target=targets_batch).detach().item()
        elif acc_fcn_type == "sklearn":
            targets_hat = torch.argmax(logits_batch, dim=1)
            y_pred = targets_hat.to(device="cpu")
            y_true = targets_batch.to(device="cpu")
            _acc += acc_fcn(y_pred=y_pred, y_true=y_true)

    return _loss, _acc


def _evaluate_epoch(model,
                    dataloader_val,
                    device,
                    move_data_to_device,
                    loss_fcn,
                    acc_fcn,
                    acc_fcn_type="torcheval"):
    """
    Evaluates the model for one epoch on the validation dataset.
    
    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader_val (torch.utils.data.DataLoader): Dataloader for the validation set.
        device (str or torch.device): Device on which to run the evaluation.
        move_data_to_device (bool): Whether to move data and targets to device.
        loss_fcn (callable): Loss function for computing validation loss.
        acc_fcn (callable): Accuracy function.
        acc_fcn_type (str): Accuracy function type ("torcheval" or "sklearn").
    
    Returns:
        tuple: (accumulated_loss, accumulated_accuracy) for the epoch.
    """
    model = model.to(device=device)
    model.eval()
    with torch.inference_mode():
        _loss, _acc = 0, 0
        for data_batch, targets_batch in dataloader_val:
            if move_data_to_device:
                data_batch = data_batch.to(device=device)
                targets_batch = targets_batch.to(device=device)

            logits_batch = model(data_batch)

            _loss += loss_fcn(input=logits_batch, target=targets_batch).detach().item()

            if acc_fcn_type == "torcheval":
                _acc += acc_fcn(input=logits_batch, target=targets_batch).detach().item()
            elif acc_fcn_type == "sklearn":
                targets_hat = torch.argmax(logits_batch, dim=1)
                y_pred = targets_hat.to(device="cpu")
                y_true = targets_batch.to(device="cpu")
                _acc += acc_fcn(y_pred=y_pred, y_true=y_true)

    return _loss, _acc


def train_model(model,
                dataloader_tr,
                dataloader_val,
                n_epochs,
                optim_fcn,
                loss_fcn,
                acc_fcn,
                acc_fcn_type="torcheval",
                device="",
                move_data_to_device=True,
                logging_interval=5,
                run_wandb=False,
                wandb_api_key="",
                wandb_notebook_name="",
                project_name="",
                run_note="",
                clip_grad=False,
                all_config=None):
    """
    Trains a PyTorch model with optional Weights & Biases (W&B) integration.
    
    Args:
        model (torch.nn.Module): The neural network model to train.
        dataloader_tr (torch.utils.data.DataLoader): Training dataloader.
        dataloader_val (torch.utils.data.DataLoader): Validation dataloader.
        n_epochs (int): Number of training epochs.
        optim_fcn (torch.optim.Optimizer): Optimizer.
        loss_fcn (callable): Loss function.
        acc_fcn (callable): Accuracy function.
        acc_fcn_type (str): Type of accuracy function ("torcheval" or "sklearn"). Defaults to "torcheval".
        device (str): Device for training ("cpu" or "cuda").
        move_data_to_device (bool): Whether to move data to device. Set False for dict inputs (LF/DF).
        logging_interval (int): Epoch interval for printing results. Defaults to 5.
        run_wandb (bool): Enable W&B logging. Defaults to False.
        wandb_api_key (str): W&B API key (can also be set via WANDB_API_KEY env var).
        wandb_notebook_name (str): Name for the W&B notebook log.
        project_name (str): W&B project name.
        run_note (str): Additional notes for the W&B run.
        clip_grad (bool): Whether to apply gradient clipping. Defaults to False.
        all_config (dict, optional): Configuration dictionary to log to W&B.
    
    Returns:
        tuple: (report, wandb_info)
            - report (dict): Training/validation losses and accuracies with keys
              'loss_tr', 'loss_val', 'acc_tr', 'acc_val'.
            - wandb_info (dict or None): W&B run info if run_wandb is True, else None.
    """
    if run_wandb:
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        if wandb_notebook_name:
            os.environ["WANDB_NOTEBOOK_NAME"] = wandb_notebook_name
        run_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        wandb.login()
        wandb.init(project=project_name,
                   name=run_name,
                   notes=run_note,
                   config=all_config)

    loss_tr, acc_tr = [], []
    loss_val, acc_val = [], []

    for epoch in range(n_epochs):
        _loss, _acc = _train_epoch(model=model,
                                   dataloader_tr=dataloader_tr,
                                   device=device,
                                   move_data_to_device=move_data_to_device,
                                   optim_fcn=optim_fcn,
                                   loss_fcn=loss_fcn,
                                   acc_fcn=acc_fcn,
                                   clip_grad=clip_grad,
                                   acc_fcn_type=acc_fcn_type)
        loss_tr.append(_loss / len(dataloader_tr))
        acc_tr.append(_acc / len(dataloader_tr))

        _loss, _acc = _evaluate_epoch(model=model,
                                      dataloader_val=dataloader_val,
                                      device=device,
                                      move_data_to_device=move_data_to_device,
                                      loss_fcn=loss_fcn,
                                      acc_fcn=acc_fcn,
                                      acc_fcn_type=acc_fcn_type)
        loss_val.append(_loss / len(dataloader_val))
        acc_val.append(_acc / len(dataloader_val))
        
        if epoch % logging_interval == 0 or epoch == n_epochs - 1:
            print(f"\nepoch{epoch}: loss_tr={loss_tr[-1]:.4f} | accuracy_tr={acc_tr[-1]:.4f}")
            print(f"epoch{epoch}: loss_val={loss_val[-1]:.4f} | accuracy_val={acc_val[-1]:.4f}")

        if run_wandb:
            wandb.log({
                "epoch": epoch,
                "loss_tr": loss_tr[-1],
                "acc_tr": acc_tr[-1],
                "loss_val": loss_val[-1],
                "acc_val": acc_val[-1],
            })

    report = {
        "loss_tr": loss_tr,
        "acc_tr": acc_tr,
        "loss_val": loss_val,
        "acc_val": acc_val,
    }

    if run_wandb:
        wandb_info = {"project_name": project_name, "run_name": run_name}
        wandb.finish()
        return report, wandb_info
    
    return report, None


def save_checkpoint(model,
                    data_config,
                    model_config,
                    train_config,
                    wandb_info,
                    checkpoints_dir):
    """
    Saves the model checkpoint along with all configurations.

    Args:
        model (torch.nn.Module): The trained model.
        data_config (dict): Data configuration settings.
        model_config (dict): Model hyperparameters.
        train_config (dict): Training configuration.
        wandb_info (dict): W&B run info with keys "project_name" and "run_name".
        checkpoints_dir (str): Parent directory for saving checkpoints.

    Raises:
        FileNotFoundError: If `checkpoints_dir` does not exist.
    """
    if not Path(checkpoints_dir).is_dir():
        raise FileNotFoundError(f"Parent directory {checkpoints_dir} does not exist")

    project_dir = os.path.join(checkpoints_dir, wandb_info["project_name"])
    if not Path(project_dir).is_dir():
        Path(project_dir).mkdir(parents=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "data_config": data_config,
        "model_config": model_config,
        "train_config": train_config,
        "wandb_info": wandb_info,
    }

    model_path = os.path.join(project_dir, f"checkpoint_{wandb_info['run_name']}.pth")
    torch.save(checkpoint, model_path)
    print(f"Checkpoint saved to {model_path}")
