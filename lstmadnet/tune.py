import os
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from lstmadnet.nn import LSTN


def setup_sweep(sweep_config, project_name, wandb_api_key="", wandb_notebook_name=""):
    """
    Sets up a W&B sweep for hyperparameter tuning.

    Args:
        sweep_config (dict): Sweep configuration defining parameters and search strategy.
        project_name (str): W&B project name.
        wandb_api_key (str, optional): W&B API key (can also be set via env var).
        wandb_notebook_name (str, optional): Notebook name for W&B logging.

    Returns:
        str: The unique sweep ID.
    """
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    if wandb_notebook_name:
        os.environ["WANDB_NOTEBOOK_NAME"] = wandb_notebook_name
    wandb.login()
    sweep_id = wandb.sweep(project=project_name, sweep=sweep_config)
    return sweep_id


def load_data(sweep_config, dataset_tr, dataset_val):
    """
    Creates DataLoaders from datasets using the sweep's batch_size.

    Args:
        sweep_config: W&B config object with a "batch_size" key.
        dataset_tr: Training dataset.
        dataset_val: Validation dataset.

    Returns:
        tuple: (dataloader_tr, dataloader_val)
    """
    dataloader_tr = DataLoader(dataset=dataset_tr,
                               batch_size=sweep_config["batch_size"],
                               shuffle=True,
                               drop_last=True)
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=sweep_config["batch_size"],
                                shuffle=False,
                                drop_last=False)
    return dataloader_tr, dataloader_val


def create_model(sweep_config, model_config):
    """
    Creates an LSTN model using sweep hyperparameters merged with base model config.

    Args:
        sweep_config: W&B config object with LSTLayer hyperparameters.
        model_config (dict): Base model configuration.

    Returns:
        LSTN: The initialized model.
    """
    _model_config = deepcopy(model_config)
    _model_config["stlayer_config"]["kernel_ratio"] = sweep_config["stlayer_config_kernel_ratio"]
    _model_config["stlayer_config"]["stride_ratio"] = sweep_config["stlayer_config_stride_ratio"]
    _model_config["stlayer_config"]["kernel_ratio_lp"] = sweep_config["stlayer_config_kernel_ratio_lp"]
    _model_config["stlayer_config"]["stride_ratio_lp"] = sweep_config["stlayer_config_stride_ratio_lp"]

    model = LSTN(**_model_config)
    return model


def set_up_training(sweep_config, model):
    """
    Creates an optimizer based on sweep hyperparameters.

    Args:
        sweep_config: W&B config with a "lr" key.
        model (torch.nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: SGD optimizer.
    """
    optim_fcn = optim.SGD(params=model.parameters(), lr=sweep_config["lr"])
    return optim_fcn


def _train_epoch(model, dataloader_tr, device, optim_fcn, loss_fcn, acc_fcn):
    model = model.to(device=device)
    model.train()
    _loss, _acc = 0, 0
    for data_batch, targets_batch in dataloader_tr:
        data_batch = data_batch.to(device=device)
        targets_batch = targets_batch.to(device=device)

        optim_fcn.zero_grad()
        logits_batch = model(data_batch)

        loss_batch = loss_fcn(input=logits_batch, target=targets_batch)
        loss_batch.backward()
        optim_fcn.step()

        _loss += loss_batch.detach().item()
        _acc += acc_fcn(input=logits_batch, target=targets_batch).detach().item()

    return _loss, _acc


def _evaluate_epoch(model, dataloader_val, device, loss_fcn, acc_fcn):
    model = model.to(device=device)
    model.eval()
    with torch.inference_mode():
        _loss, _acc = 0, 0
        for data_batch, targets_batch in dataloader_val:
            data_batch = data_batch.to(device=device)
            targets_batch = targets_batch.to(device=device)

            logits_batch = model(data_batch)

            _loss += loss_fcn(input=logits_batch, target=targets_batch).detach().item()
            _acc += acc_fcn(input=logits_batch, target=targets_batch).detach().item()

    return _loss, _acc


def train_model(model, dataloader_tr, dataloader_val, n_epochs, optim_fcn, loss_fcn, acc_fcn, device):
    """
    Trains a model during a sweep run and logs metrics to W&B.

    Args:
        model (torch.nn.Module): Model to train.
        dataloader_tr: Training dataloader.
        dataloader_val: Validation dataloader.
        n_epochs (int): Number of epochs.
        optim_fcn: Optimizer.
        loss_fcn: Loss function.
        acc_fcn: Accuracy function.
        device: Training device.
    """
    for epoch in range(n_epochs):
        _loss, _acc = _train_epoch(model=model,
                                   dataloader_tr=dataloader_tr,
                                   device=device,
                                   optim_fcn=optim_fcn,
                                   loss_fcn=loss_fcn,
                                   acc_fcn=acc_fcn)
        loss_tr = _loss / len(dataloader_tr)
        acc_tr = _acc / len(dataloader_tr)

        _loss, _acc = _evaluate_epoch(model=model,
                                      dataloader_val=dataloader_val,
                                      device=device,
                                      loss_fcn=loss_fcn,
                                      acc_fcn=acc_fcn)
        loss_val = _loss / len(dataloader_val)
        acc_val = _acc / len(dataloader_val)

        wandb.log({
            "epoch": epoch,
            "loss_tr": loss_tr,
            "acc_tr": acc_tr,
            "loss_val": loss_val,
            "acc_val": acc_val,
        })


def agent_function(sweep_id, dataset_tr, dataset_val, model_config, n_epochs, loss_fcn, acc_fcn, device):
    """
    Function to pass to `wandb.agent` for executing hyperparameter sweep runs.

    Each call initializes a W&B run, reads sweep hyperparameters, builds a model,
    trains it, and logs results.

    Args:
        sweep_id (str): The W&B sweep ID.
        dataset_tr (Dataset): Training dataset.
        dataset_val (Dataset): Validation dataset.
        model_config (dict): Base model configuration.
        n_epochs (int): Number of training epochs per run.
        loss_fcn: Loss function.
        acc_fcn: Accuracy function.
        device: Training device.
    """
    run_name = f"{sweep_id}_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    wandb.init(name=run_name)

    sweep_config = wandb.config

    dataloader_tr, dataloader_val = load_data(sweep_config=sweep_config,
                                              dataset_tr=dataset_tr,
                                              dataset_val=dataset_val)

    model = create_model(sweep_config=sweep_config, model_config=model_config)

    optim_fcn = set_up_training(sweep_config, model=model)

    train_model(model=model,
                dataloader_tr=dataloader_tr,
                dataloader_val=dataloader_val,
                n_epochs=n_epochs,
                optim_fcn=optim_fcn,
                loss_fcn=loss_fcn,
                acc_fcn=acc_fcn,
                device=device)

    wandb.finish()


def save_sweep_checkpoint(sweep_checkpoint, sweeps_dir, project_name, sweep_id):
    """
    Saves a sweep checkpoint (best model config and weights).

    Args:
        sweep_checkpoint (dict): Checkpoint data to save.
        sweeps_dir (str): Base directory for sweep outputs.
        project_name (str): W&B project name.
        sweep_id (str): The sweep ID.
    """
    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    target_dir = os.path.join(sweeps_dir, project_name, f"{now}_{sweep_id}")
    if not Path(target_dir).is_dir():
        Path(target_dir).mkdir(parents=True)
    
    checkpoint_path = os.path.join(target_dir, f"{sweep_id}_checkpoint.pth")
    torch.save(sweep_checkpoint, checkpoint_path)
    print(f"Sweep checkpoint saved to {checkpoint_path}")
