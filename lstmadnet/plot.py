import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_signal(signal,
                figsize=(4, 2),
                alpha=1,
                xlabel=None,
                ylabel=None,
                label="signal",
                title=None,
                show=False):
    """
    Plots a 1D signal stored in a torch.Tensor.

    Args:
        signal (torch.Tensor): A 1D tensor representing the signal.
        figsize (tuple, optional): Figure size (width, height). Default is (4, 2).
        alpha (float, optional): Line transparency. Default is 1.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        label (str, optional): Legend label. Default is "signal".
        title (str, optional): Plot title.
        show (bool, optional): If True, display the plot. Default is False.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Raises:
        ValueError: If the input is not a 1D torch.Tensor.
    """
    if not isinstance(signal, torch.Tensor) or signal.ndim != 1:
        raise ValueError("Input signal should be a 1D torch.Tensor object")
    if signal.device.type == "cuda":
        signal = signal.to(device="cpu")

    signal = signal.detach().numpy()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    if np.any(np.imag(signal)):
        ax.plot(np.real(signal), alpha=alpha, label=f"{label} - Real")
        ax.plot(np.imag(signal), alpha=alpha, label=f"{label} - Imaginary")    
    else:
        ax.plot(signal, alpha=alpha, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if show:
        plt.show()
    return fig


def plot_sample(sample_intrinsic,
                sample_task,
                sample_extrinsic,
                figsize=(12, 6),
                wspace=0.15,
                hspace=0.4,
                show=False):
    """
    Plots multi-modal data for a single sample from the TriScrewSense dataset.

    Args:
        sample_intrinsic (pd.DataFrame): Intrinsic sensor data (torque, current vs angle/depth).
        sample_task (pd.DataFrame): Task-related data (TCP positions vs time).
        sample_extrinsic (pd.DataFrame): Extrinsic sensor data (sound vs time).
        figsize (tuple, optional): Figure size. Default is (12, 6).
        wspace (float, optional): Width space between subplots. Default is 0.15.
        hspace (float, optional): Height space between subplots. Default is 0.4.
        show (bool, optional): If True, display the plot. Default is False.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    sns.set_style("whitegrid")
    sample_intrinsic.plot(ax=axs[0, 0], x="angle", y=["torque", "current"])
    sample_intrinsic.plot(ax=axs[0, 1], x="depth", y=["torque", "current"])
    sample_task.plot(ax=axs[1, 0], x="time", y=["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "current"])
    sample_extrinsic.plot(ax=axs[1, 1], x="time", y=["sound"], alpha=0.7)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.suptitle(f"label={sample_intrinsic.label.unique()[0]}")
    if show:
        plt.show()
    return fig


def plot_tfr(tfr,
             figsize=(4, 3),
             xlabel=None,
             ylabel=None,
             title="TFR",
             cbar_title=None,
             show=False):
    """
    Plots a time-frequency representation (TFR) stored in a 2D torch.Tensor.

    Args:
        tfr (torch.Tensor): A 2D tensor representing the TFR.
        figsize (tuple, optional): Figure size. Default is (4, 3).
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        title (str, optional): Plot title. Default is "TFR".
        cbar_title (str, optional): Colorbar title.
        show (bool, optional): If True, display the plot. Default is False.

    Returns:
        matplotlib.figure.Figure: The figure object.

    Raises:
        ValueError: If the input is not a 2D torch.Tensor.
    """
    if not isinstance(tfr, torch.Tensor) or tfr.ndim != 2:
        raise ValueError("tfr should be a 2D torch.Tensor object")
    if tfr.device.type == "cuda":
        tfr = tfr.to(device="cpu")
    
    tfr = tfr.detach().numpy()

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    ah = ax.imshow(tfr, aspect="auto", origin="lower", cmap="inferno")
    fig.colorbar(ah, label=cbar_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show:
        plt.show()
    return fig


def plot_report(report,
                figsize=(8, 3),
                show=False):
    """
    Plots training and validation loss and accuracy curves.

    Args:
        report (dict): Training report with keys 'loss_tr', 'loss_val', 'acc_tr', 'acc_val'.
        figsize (tuple, optional): Figure size. Default is (8, 3).
        show (bool, optional): If True, display the plot. Default is False.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].plot(report["loss_tr"], label="training")
    axs[0].plot(report["loss_val"], label="validation")
    axs[0].set_xlabel("epoch")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(report["acc_tr"], label="training")
    axs[1].plot(report["acc_val"], label="validation")
    axs[1].set_xlabel("epoch")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    if show:
        plt.show()
    return fig
