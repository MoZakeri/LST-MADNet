import os
import re
import json
from pathlib import Path

import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def load_raw_data(raw_path, thr=-20, sr=None):
    """
    Loads and processes raw data from a directory containing intrinsic (.json),
    task (.csv), and extrinsic (.wav) measurements.

    Args:
        raw_path (str): Path to the directory containing raw data files.
        thr (int, optional): Threshold index for slicing intrinsic measurements.
            Must be a negative integer. Default is -20.
        sr (int, optional): Sampling rate for audio data. If None, uses librosa default.

    Returns:
        tuple: Three DataFrames (dataset_intrinsic, dataset_task, dataset_extrinsic).

    Raises:
        ValueError: If `thr` is not a negative integer.
    """
    if not isinstance(thr, int) or thr > 0:
        raise ValueError("thr should be a negative integer")
    
    labels = dict(
        A=0,  # normal
        B=1,  # under-tightening
        C=2,  # over-tightening
        N=3,  # missing screw
        S=4,  # pose anomaly (excluded)
    )
    dataset_intrinsic_list = []
    dataset_task_list = []
    dataset_extrinsic_list = []
    id_int = 0
    id_task = 0
    id_ext = 0

    for dir_name in tqdm(os.listdir(raw_path), desc="Loading files"):
        dir_path = os.path.join(raw_path, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            process_id = file_name[:file_name.find(".")]
            label = labels[file_name[re.search("[a-zA-Z]", file_name).start()]]
            if label == 4:
                print(f"Skipping {file_path} (pose anomaly, label={label})")
                continue
            
            if file_name.endswith(".json"):
                with open(file_path, "r") as f:
                    data_ = json.load(f)["XML_Data"]["Wsk3Vectors"]["Y_AxesList"]["AxisData"]
                n_set_ = np.array(data_[0]["Values"]["float"], dtype=np.float64)
                torque_ = np.array(data_[1]["Values"]["float"], dtype=np.float64)
                current_ = np.array(data_[2]["Values"]["float"], dtype=np.float64)
                angle_ = np.array(data_[3]["Values"]["float"], dtype=np.float64)
                depth_ = np.array(data_[4]["Values"]["float"], dtype=np.float64)
                
                data_intrinsic = pd.DataFrame()
                data_intrinsic["id"] = None
                data_intrinsic["process_id"] = None
                data_intrinsic["n_set"] = n_set_[:thr]
                data_intrinsic["torque"] = torque_[:thr]
                data_intrinsic["current"] = current_[:thr]
                data_intrinsic["angle"] = angle_[:thr] * (np.pi / 180)
                data_intrinsic["depth"] = depth_[:thr]
                data_intrinsic["label"] = label
                data_intrinsic["id"] = id_int
                data_intrinsic["process_id"] = process_id

                dataset_intrinsic_list.append(data_intrinsic)
                id_int += 1

            if file_name.endswith(".csv"):
                data_ = pd.read_csv(file_path)

                data_task = pd.DataFrame()
                data_task["id"] = None
                data_task["process_id"] = None
                data_task["time"] = data_["Time (ms)"]
                data_task["tcp_x"] = data_["TCP_x (mm)"]
                data_task["tcp_y"] = data_["TCP_y (mm)"]
                data_task["tcp_z"] = data_["TCP_z (mm)"]
                data_task["tcp_rx"] = data_["TCP_rx (mm)"]
                data_task["tcp_ry"] = data_["TCP_ry (mm)"]
                data_task["tcp_rz"] = data_["TCP_rz (mm)"]
                data_task["current"] = data_["Robot_I (A)"]
                data_task["label"] = label
                data_task["id"] = id_task
                data_task["process_id"] = process_id

                dataset_task_list.append(data_task)
                id_task += 1

            if file_name.endswith(".wav"):
                data_, sr = librosa.load(file_path, sr=sr)
                time_ = np.linspace(0, data_.size / sr, data_.size) * 1000

                data_extrinsic = pd.DataFrame()
                data_extrinsic["id"] = None
                data_extrinsic["process_id"] = None
                data_extrinsic["time"] = time_
                data_extrinsic["sound"] = data_
                data_extrinsic["label"] = label
                data_extrinsic["id"] = id_ext
                data_extrinsic["process_id"] = process_id

                dataset_extrinsic_list.append(data_extrinsic)
                id_ext += 1
    
    dataset_intrinsic = pd.concat(dataset_intrinsic_list, ignore_index=True)
    dataset_task = pd.concat(dataset_task_list, ignore_index=True)
    dataset_extrinsic = pd.concat(dataset_extrinsic_list, ignore_index=True)
    
    for name, ds in [("intrinsic", dataset_intrinsic), ("task", dataset_task), ("extrinsic", dataset_extrinsic)]:
        labels_arr = ds.groupby("id").label.unique()
        vals, cnts = np.unique(labels_arr, return_counts=True)
        print(f"{name}: classes={vals} | counts={cnts}")
    
    return dataset_intrinsic, dataset_task, dataset_extrinsic


def save_dataset(dataset, target_path, dataset_name="dataset"):
    """
    Saves a DataFrame to a Parquet file.

    Args:
        dataset (pd.DataFrame): The dataset to save.
        target_path (str): Directory where the file will be saved.
        dataset_name (str, optional): Filename without extension. Default is "dataset".

    Raises:
        FileNotFoundError: If `target_path` does not exist.
    """
    if not Path(target_path).is_dir():
        raise FileNotFoundError(f"Parent directory {target_path} does not exist")

    print(f"Saving {dataset_name} in {target_path} ...")
    dataset.to_parquet(os.path.join(target_path, f"{dataset_name}.parquet"), engine="fastparquet")


def class_distribution(dataset):
    """
    Computes the class distribution within a dataset.

    Args:
        dataset (pd.DataFrame): Dataset with 'label' and 'id' columns.

    Returns:
        tuple: (labels, counts, distributions) as numpy arrays.
    """
    labels_all = dataset.groupby("id")["label"].apply(lambda ds: ds.unique()[0])
    labels, cnts = np.unique(labels_all, return_counts=True)
    dists = cnts / cnts.sum()
    return labels, cnts, dists


def split_dataset(dataset_path, tr_ratio, random_state=42):
    """
    Splits a dataset into stratified training and validation sets.

    Args:
        dataset_path (str): Path to the Parquet file.
        tr_ratio (float): Proportion for the training set (e.g., 0.7).
        random_state (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (dataset_tr, dataset_val) as DataFrames.
    """
    print(f"Loading the dataset from {dataset_path} ...")
    dataset = pd.read_parquet(dataset_path, engine="pyarrow")

    print("Splitting the dataset ...")
    idxs = dataset.groupby("id")["id"].apply(lambda df: df.unique()[0])
    labels = dataset.groupby("id")["label"].apply(lambda df: df.unique()[0])
    idxs_tr, idxs_val = train_test_split(idxs,
                                         stratify=labels,
                                         shuffle=True,
                                         train_size=tr_ratio,
                                         random_state=random_state)
    
    dataset_tr = dataset[dataset["id"].isin(idxs_tr)].reset_index(drop=True)
    dataset_val = dataset[dataset["id"].isin(idxs_val)].reset_index(drop=True)
    return dataset_tr, dataset_val


class TriScrewSense(Dataset):
    """
    PyTorch Dataset for single-feature time-series data from the TriScrewSense dataset.

    Loads a Parquet file, extracts a single feature, resamples to a fixed length,
    and optionally normalizes the data.

    Args:
        dataset_path (str): Path to the dataset Parquet file.
        target_feature (str): Feature column to extract (e.g., "torque", "sound").
        target_length (int, optional): Fixed length after resampling. If None, determined by resample_mode.
        resample_mode (str, optional): How to determine target_length: "min", "mean", or "max". Default is "min".
        normalize (bool, optional): Whether to normalize data. Default is False.
        mean (torch.Tensor, optional): Pre-computed mean for normalization.
        std (torch.Tensor, optional): Pre-computed std for normalization.

    Attributes:
        data (torch.Tensor): Processed data of shape (num_samples, 1, target_length).
        targets (torch.Tensor): Labels of shape (num_samples,).
        mean (torch.Tensor): Mean used for normalization (computed if not provided).
        std (torch.Tensor): Std used for normalization (computed if not provided).

    Notes:
        - Minimum target lengths: intrinsic=1408, task=463, extrinsic=34816.
    """
    def __init__(self,
                 dataset_path,
                 target_feature,
                 target_length=None,
                 resample_mode="min",
                 normalize=False,
                 mean=None,
                 std=None):
        
        super().__init__()
        self._validate_inputs(dataset_path, target_length, resample_mode)
        
        print(f"Loading dataset from {dataset_path} ...")
        self.dataset = pd.read_parquet(dataset_path, engine="pyarrow")
        self.target_feature = target_feature
        self.resample_mode = resample_mode
        self.target_length = self._get_target_length() if target_length is None else target_length
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.data, self.targets = self._extract_feature()

    def _validate_inputs(self, dataset_path, target_length, resample_mode):
        valid_resample_modes = ["min", "mean", "max"]

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"No dataset was found at {dataset_path}")
        if (target_length is not None) and (not isinstance(target_length, int) or target_length <= 0):
            raise ValueError("target_length should be a positive integer")
        if resample_mode not in valid_resample_modes:
            raise ValueError(f"resample_mode should be in {valid_resample_modes}")

    def _get_target_length(self):
        print("Computing target length ...")
        resample_map = dict(min=np.min, mean=np.mean, max=np.max)
        resample_fcn = resample_map[self.resample_mode]
        lengths = self.dataset.groupby("id").size()
        return int(resample_fcn(lengths))
    
    def _extract_feature(self):
        """Extracts and resamples the target feature into tensors."""
        valid_target_features = set(self.dataset.columns) - {"id", "process_id", "time", "label"}
        if self.target_feature not in valid_target_features:
            raise ValueError(f"Invalid target_feature='{self.target_feature}', "
                             f"should be one of {valid_target_features}")
        
        samples = self.dataset.groupby("id")
        data = torch.empty(size=(len(samples), self.target_length), dtype=torch.float)
        targets = torch.empty(size=(len(samples),), dtype=torch.long)
        for i, (_, sample) in tqdm(enumerate(samples), 
                                   total=len(samples), 
                                   desc=f"Extracting '{self.target_feature}', target_length={self.target_length}"):
            feature = resample(getattr(sample, self.target_feature).to_numpy(), num=self.target_length)
            label = sample["label"].unique()[0]
            data[i, :] = torch.tensor(feature)
            targets[i] = label
        data = data.reshape(data.shape[0], 1, data.shape[-1])

        if self.normalize and (self.mean is None or self.std is None):
            self.mean = data.mean(dim=0, keepdim=True)
            self.std = data.std(dim=0, keepdim=True) + 1e-7
            data = (data - self.mean) / self.std
        elif self.normalize:
            data = (data - self.mean) / self.std

        return data, targets
    
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx, :], self.targets[idx]


class TriScrewSenseEF(Dataset):
    """
    PyTorch Dataset for Early Fusion (EF) on the TriScrewSense dataset.

    Loads all channels from all modalities, normalizes each independently,
    and concatenates them along the time dimension.

    Args:
        data_dir (str): Directory containing the dataset Parquet files.
        training (bool): If True, load training data; otherwise validation data.
        mean (torch.Tensor, optional): Mean for normalization (required for validation).
        std (torch.Tensor, optional): Std for normalization (required for validation).

    Attributes:
        data (torch.Tensor): Fused input tensor of shape (num_samples, 1, total_length).
        targets (torch.Tensor): Labels of shape (num_samples,).
        mean (torch.Tensor): Mean of the fused training data.
        std (torch.Tensor): Std of the fused training data.
    """
    def __init__(self,
                 data_dir,
                 training=True,
                 mean=None,
                 std=None):

        super().__init__()

        target_features = {
            "intrinsic": ["angle", "depth", "torque", "current"],
            "task": ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "current"],
            "extrinsic": ["sound"]
        }
        target_lengths = {
            "intrinsic": 1408,
            "task": 463,
            "extrinsic": 34816
        }

        data_list = []
        _read_targets = False
        for _file_name in os.listdir(data_dir):
            if training:
                if _file_name.find("_tr") == -1:
                    continue
            else:
                if _file_name.find("_val") == -1:
                    continue

            _file_path = os.path.join(data_dir, _file_name)
            _data_type = _file_name[:_file_name.find("_")]
            if _data_type not in target_features:
                continue
            _target_features = target_features[_data_type]

            for _target_feature in _target_features:
                _dataset = TriScrewSense(dataset_path=_file_path,
                                         target_feature=_target_feature,
                                         target_length=target_lengths[_data_type],
                                         normalize=True)
                data_list.append(_dataset.data)
                if not _read_targets:
                    self.targets = _dataset.targets
                    _read_targets = True
        _data = torch.cat(data_list, dim=-1)

        if training:
            self.data = _data
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)
        else:
            self.data = (_data - mean) / std
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TriScrewSenseLF(Dataset):
    """
    PyTorch Dataset for Late Fusion (LF) and Deep Fusion (DF) on the TriScrewSense dataset.

    Returns per-sample dictionaries mapping channel names to their respective tensors.
    Compatible with both LSTMADNetLF and LSTMADNetDF architectures.

    Args:
        data_dir (str): Directory containing the dataset Parquet files.
        training (bool): If True, load training data; otherwise validation data.
        target_features (dict): Maps modality names to lists of feature columns.
            Default: {"intrinsic": ["torque"], "task": ["tcp_x"], "extrinsic": ["sound"]}.
        device (str): Device for tensors ('cpu' or 'cuda'). Default is 'cpu'.

    Attributes:
        data (list[dict]): Per-sample dicts mapping channel name to tensor of shape (1, signal_length).
        targets (list[torch.Tensor]): Per-sample label tensors.
        channels (list[str]): List of channel names in the dataset.
    """
    def __init__(self,
                 data_dir,
                 training=True,
                 target_features=None,
                 device="cpu"):
        super().__init__()

        if target_features is None:
            target_features = {
                "intrinsic": ["torque"],
                "task": ["tcp_x"],
                "extrinsic": ["sound"]
            }

        print("Creating data and targets buffers ...")
        file_name = "task_tr.parquet" if training else "task_val.parquet"
        dataset = pd.read_parquet(os.path.join(data_dir, file_name))
        self.size = dataset.id.nunique()
        print(f"Dataset size: {self.size}")
        self.data = [{} for _ in range(self.size)]
        self.targets = [None for _ in range(self.size)]

        exclude_columns = {"id", "process_id", "time", "n_set", "label"}
        target_lengths = {
            "intrinsic": 1408,
            "task": 463,
            "extrinsic": 34816
        }
        target_data_types = list(target_features.keys())
        first_read = True

        self.channels = []
        for file_name in os.listdir(data_dir):
            data_type = file_name[0:file_name.find("_")]
            if data_type not in target_data_types:
                continue

            if training:
                if file_name.find("_tr") == -1:
                    continue
            else:
                if file_name.find("_val") == -1:
                    continue

            print(f"\nLoading dataset from {file_name} ...")
            file_path = os.path.join(data_dir, file_name)
            dataset = pd.read_parquet(file_path)

            target_columns = set(target_features[data_type]) - exclude_columns
            self.channels.extend(list(target_columns))
            for idd, sample in tqdm(enumerate(dataset.groupby("id")),
                                    total=dataset.id.nunique(),
                                    desc=f"Extracting {target_columns}"):
                sample = sample[1]
                for column in target_columns:
                    feature = resample(sample[column].to_numpy(), num=target_lengths[data_type])
                    self.data[idd][column] = torch.tensor(feature,
                                                          dtype=torch.float,
                                                          device=device).unsqueeze(dim=0)
                if first_read:
                    self.targets[idd] = torch.tensor(
                        sample.label.unique()[0], dtype=torch.long, device=device)

            first_read = False

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
