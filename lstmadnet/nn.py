import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fcn


class LSTLayer1D(nn.Module):
    """
    A 1D Learnable Scattering Transform (LST) layer for processing time-series data. This layer applies a 
    wavelet transform followed by a modulus and low-pass filtering operation to extract time-frequency representation 
    from the input signal.

    Args:
        scale_min (float): The minimum scale value for the wavelet transformation.
        scale_max (float): The maximum scale value for the wavelet transformation.
        num_scales (int): The number of scales to generate between `scale_min` and `scale_max`.
        kernel_ratio (float): Ratio to determine the wavelet transform kernel size relative to the input signal size.
        stride_ratio (float): Ratio to determine the wavelet transform stride size relative to its kernel size.
        kernel_ratio_lp (float): Ratio to determine the low-pass filtering kernel size relative to the modulus response signal size.
        stride_ratio_lp (float): Ratio to determine the low pass filtering stride size relative to its kernel size.

    Attributes:
        scale_min (torch.Tensor): Tensor representation of the minimum scale.
        scale_max (torch.Tensor): Tensor representation of the maximum scale.
        num_scales (int): The number of scales used for the scattering transform.
        kernel_ratio (float): Ratio to determine the wavelet transform kernel size relative to the input signal size.
        stride_ratio (float): Ratio to determine the wavelet transform stride size relative to its kernel size.
        kernel_ratio_lp (float): Ratio to determine the low-pass filtering kernel size relative to the modulus response signal size.
        stride_ratio_lp (float): Ratio to determine the low pass filtering stride size relative to its kernel size.
        scales (torch.nn.Parameter): Parameterized scales generated between `scale_min` and `scale_max`.
        wavelet_kernel_real (torch.Tensor): Real part of the wavelet kernel used in the wavelet transform.
        wavelet_kernel_imag (torch.Tensor): Imaginary part of the wavelet kernel used in the wavelet transform.
        wavelet_response_real (torch.Tensor): Real part of the wavelet response after convolution.
        wavelet_response_imag (torch.Tensor): Imaginary part of the wavelet response after convolution.
        modulus_response (torch.Tensor): Modulus of the wavelet response.
        gaussian_kernel (torch.Tensor): The Gaussian kernel used for low-pass filtering.
        filter_response (torch.Tensor): The final output after low-pass filtering.

    Methods:
        reset(): Resets the internal states including kernels and responses.
        forward(x): Applies the wavelet transform, modulus operation, and low-pass filtering to the input tensor.
    """
    def __init__(self,
                 scale_min,
                 scale_max,
                 num_scales,
                 kernel_ratio,
                 stride_ratio,
                 kernel_ratio_lp,
                 stride_ratio_lp):
        
        super().__init__()
        self._validate_inputs(scale_min,
                              scale_max,
                              num_scales,
                              kernel_ratio,
                              stride_ratio,
                              kernel_ratio_lp,
                              stride_ratio_lp)
        
        self.scale_min = torch.tensor(scale_min, dtype=torch.float)
        self.scale_max = torch.tensor(scale_max, dtype=torch.float)
        self.num_scales = num_scales
        self.kernel_ratio = kernel_ratio
        self.stride_ratio = stride_ratio
        self.kernel_ratio_lp = kernel_ratio_lp
        self.stride_ratio_lp = stride_ratio_lp

        self.scales = nn.Parameter(self._generate_scales())
        self.wavelet_kernel_real = None
        self.wavelet_kernel_imag = None
        self.wavelet_response_real = None
        self.wavelet_response_imag = None
        self.modulus_response = None
        self.gaussian_kernel = None
        self.filter_response = None

    def _is_in_range(self, n, a, b):
        return a < n < b
    
    def _validate_inputs(self,
                         scale_min,
                         scale_max,
                         num_scales,
                         kernel_ratio,
                         stride_ratio,
                         kernel_ratio_lp,
                         stride_ratio_lp):
        valid_kernel_ratio = [0, 1]
        valid_stride_ratio = [0, 1]

        if not isinstance(scale_min, (int, float)) or scale_min <= 0:
            raise ValueError("scale_min should be a positive number")
        if not isinstance(scale_max, (int, float)) or scale_max <= 0:
            raise ValueError("scale_max should be a positive number")
        if not isinstance(num_scales, int) or num_scales <= 0:
            raise ValueError("num_scales should be a positive integer")
        
        if not isinstance(kernel_ratio, float) or not self._is_in_range(kernel_ratio, valid_kernel_ratio[0], valid_kernel_ratio[1]):
            raise ValueError(f"kernel_ratio should be a number in {valid_kernel_ratio}")
        
        if not isinstance(stride_ratio, float) or not self._is_in_range(stride_ratio, valid_stride_ratio[0], valid_stride_ratio[1]):
            raise ValueError(f"stride_ratio should be a number in {valid_stride_ratio}")
        
        if not isinstance(kernel_ratio_lp, float) or not self._is_in_range(kernel_ratio_lp, valid_kernel_ratio[0], valid_kernel_ratio[1]):
            raise ValueError(f"kernel_ratio_lp should be a number in {valid_kernel_ratio}")
        
        if not isinstance(stride_ratio_lp, float) or not self._is_in_range(stride_ratio_lp, valid_stride_ratio[0], valid_stride_ratio[1]):
            raise ValueError(f"stride_ratio_lp should be a number in {valid_stride_ratio}")

    def reset(self):
        self.wavelet_kernel = None
        self.wavelet_response = None
        self.modulus_response = None
        self.gaussian_kernel = None
        self.filter_response = None

    def __repr__(self):
        return (f"LSTLayer1D(1, {self.num_scales}, "
                f"[kernel_ratio, stride_ratio]=({self.kernel_ratio}, {self.stride_ratio}), "
                f"[kernel_ratio_lp, stride_ratio_lp]=({self.kernel_ratio_lp}, {self.stride_ratio_lp}))")

    def _generate_scales(self):
        """
        Generates a tensor of scales logarithmically spaced between `scale_min` and `scale_max`.

        Returns:
            torch.Tensor: A 1D tensor containing `num_scales` logarithmically spaced scales.
        """
        scales = torch.logspace(start=torch.log10(self.scale_min), 
                                end=torch.log10(self.scale_max), 
                                steps=self.num_scales)
        return scales

    def _morlet(self,
                kernel_size=100,
                sigma=1.,
                omega=5.,
                scale=torch.tensor(1.)):
        """
        Generates a complex Morlet wavelet and returns its real and imaginary components.  

        Args:
            kernel_size (int, optional): The size of the kernel. Default is 100.
            sigma (float, optional): The standard deviation of the Gaussian envelope. Default is 1.0.
            omega (float, optional): The frequency of the sinusoidal component of base wavelet. Default is 5.0.
            scale (torch.Tensor): The value by which the base wavelet's frequency is scaled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two 1D tensors:
                - `wavelet_real`: The real part of the Morlet wavelet.
                - `wavelet_imag`: The imaginary part of the Morlet wavelet.
        
        Raises:
            ValueError: If `scale` is not a 0D tensor.
        """
        if scale.ndim != 0:
            raise ValueError("input scale should be a 0D tensor")

        device = self.scales.device
        width = 4 * max(1, sigma)
        n = torch.linspace(-width, width, steps=int(kernel_size)).to(device)
        omega_ = omega / scale
        pi_torch = torch.tensor(np.pi)
        term_g = (1 / torch.sqrt(2 * pi_torch * sigma**2)) * torch.exp((-n**2) / (2 * sigma**2))
        term_cos = torch.cos(n * omega_)
        term_sin = torch.sin(n * omega_)
        wavelet_real = (1 / torch.sqrt(scale)) * term_g * term_cos
        wavelet_imag = (1 / torch.sqrt(scale)) * term_g * term_sin
        return wavelet_real, wavelet_imag

    def _gaussian(self,
                  filter_size=100,
                  mu=0.,
                  sigma=1.):
        """
        Generates a Gaussian kernel for low-pass filtering.

        Args:
            filter_size (int, optional): The size of the Gaussian kernel. Default is 100.
            mu (float, optional): The mean of the Gaussian. Default is 0.
            sigma (float, optional): The standard deviation of the Gaussian. Default is 1.

        Returns:
            torch.Tensor: The normalized Gaussian kernel.
        """
        device = self.scales.device
        width = 4 * max(1, sigma)
        n = torch.linspace(mu - width, mu + width, filter_size).to(device=device)
        pi_torch = torch.tensor(np.pi)
        term1 = 1 / torch.sqrt(2 * pi_torch * sigma**2)
        term2 = torch.exp((-(n - mu)**2) / (2 * sigma**2))
        signal = term1 * term2
        return signal / signal.sum()

    def _wavelet_transform(self, x):
        """
        Applies a wavelet transform to the input tensor.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, 1, signal_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - `wavelet_response_real`: Real part of the wavelet response (batch_size, num_scales, response_size).
                - `wavelet_response_imag`: Imaginary part of the wavelet response (batch_size, num_scales, response_size).
                - `wavelet_kernel_real`: Real part of the wavelet kernel (num_scales, 1, kernel_size).
                - `wavelet_kernel_imag`: Imaginary part of the wavelet kernel (num_scales, 1, kernel_size).
        """
        device = self.scales.device
        _, _, signal_size = x.shape
        kernel_size = max(int(self.kernel_ratio * signal_size), 3)
        stride_size = max(int(self.stride_ratio * kernel_size), 1)
        wavelet_kernel_real = torch.empty(size=(self.num_scales, 1, kernel_size), dtype=torch.float).to(device=device)
        wavelet_kernel_imag = torch.empty(size=(self.num_scales, 1, kernel_size), dtype=torch.float).to(device=device)
        for j, scale in enumerate(self.scales):
            wavelet_kernel_real[j, 0, :], wavelet_kernel_imag[j, 0, :] = self._morlet(kernel_size=kernel_size,
                                                                                      scale=scale)
        wavelet_response_real = fcn.conv1d(input=x,
                                           weight=wavelet_kernel_real,
                                           stride=stride_size,
                                           padding="valid")
        wavelet_response_imag = fcn.conv1d(input=x,
                                           weight=wavelet_kernel_imag,
                                           stride=stride_size,
                                           padding="valid")
        return wavelet_response_real, wavelet_response_imag, wavelet_kernel_real, wavelet_kernel_imag

    def _modulus(self, wavelet_response_real, wavelet_response_imag):
        """
        Applies the modulus operation to the wavelet response.

        Args:
            wavelet_response_real (torch.Tensor): Real part (batch_size, num_scales, response_size).
            wavelet_response_imag (torch.Tensor): Imaginary part (batch_size, num_scales, response_size).

        Returns:
            torch.Tensor: The modulus of the wavelet response (batch_size, num_scales, response_size).
        """
        modulus_response = wavelet_response_real**2 + wavelet_response_imag**2
        return modulus_response
    
    def _low_pass_filter(self, modulus_response):
        """
        Applies a low-pass filter to the modulus response.

        Args:
            modulus_response (torch.Tensor): Modulus signal (batch_size, num_scales, modulus_size).

        Returns:
            tuple: A tuple containing:
                - filter_response (torch.Tensor): Low-pass filtered signal (batch_size, num_scales, filter_size).
                - gaussian_kernel (torch.Tensor): Gaussian kernel (num_scales, 1, kernel_size).
        """
        device = self.scales.device
        _, channel_size, signal_size = modulus_response.shape
        kernel_size = max(int(self.kernel_ratio_lp * signal_size), 3)
        stride_size = max(int(self.stride_ratio * kernel_size), 1)
        gaussian_kernel = torch.empty(size=(channel_size, 1, kernel_size), dtype=torch.float).to(device=device)
        for j in range(gaussian_kernel.shape[0]):
            gaussian_kernel[j, 0, :] = self._gaussian(filter_size=kernel_size)
        filter_response = fcn.conv1d(input=modulus_response, 
                                     weight=gaussian_kernel, 
                                     stride=stride_size, 
                                     padding="valid", 
                                     groups=channel_size)
        return filter_response, gaussian_kernel
    
    def forward(self, x):
        """
        Applies the complete 1D Learnable Scattering Transform to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 1, signal_size).

        Returns:
            torch.Tensor: Filtered response with shape (batch_size, num_scales, filter_response_size).

        Raises:
            ValueError: If the input tensor is not a 3D tensor with shape (batch_size, 1, signal_size).
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("input should be a 3D tensor of shape (batch_size, channel_size=1, signal_size)")
        if x.dtype != torch.float:
            x = x.to(dtype=torch.float)

        self.wavelet_response_real, self.wavelet_response_imag, self.wavelet_kernel_real, self.wavelet_kernel_imag = self._wavelet_transform(x)
        self.modulus_response = self._modulus(self.wavelet_response_real, self.wavelet_response_imag)
        self.filter_response, self.gaussian_kernel = self._low_pass_filter(self.modulus_response)
        return self.filter_response


class LSTN(nn.Module):
    """
    Learnable Scattering Transform Network (LSTN) for single-channel time-series anomaly detection.

    Combines an LST layer with convolutional and fully connected layers for classification.

    Architecture:
        LSTLayer1D -> [BatchNorm1D] -> Conv2D -> Pool2D -> [BatchNorm2D] ->
        Conv2D -> Pool2D -> [BatchNorm2D] -> FC -> FC -> FC -> logits

    Args:
        signal_size (int): The size of the input signal.
        stlayer_config (dict): Configuration for the LSTLayer1D layer.
        conv1_config (dict): Configuration for the first Conv2D layer.
        pool1_config (dict): Configuration for the first MaxPool2D layer.
        conv2_config (dict): Configuration for the second Conv2D layer.
        pool2_config (dict): Configuration for the second MaxPool2D layer.
        linear_config (dict): Configuration for the fully connected layers.
        batch_norm (bool, optional): If True, batch normalization layers are added. Default is False.
        verbose (bool, optional): If True, prints tensor shapes after each layer. Default is False.
    """
    def __init__(self,
                 signal_size,
                 stlayer_config,
                 conv1_config,
                 pool1_config,
                 conv2_config,
                 pool2_config,
                 linear_config,
                 batch_norm=False,
                 verbose=False):
        
        super().__init__()
        self._validate_inputs(stlayer_config,
                              conv1_config,
                              pool1_config,
                              conv2_config,
                              pool2_config, 
                              linear_config)
        self._MIN_KERNEL_SIZE = 3
        self._MIN_STRIDE_SIZE = 1
        self.batch_norm = batch_norm

        if verbose:
            labels, tensors = [], []

        dummy = torch.randn(size=(1, 1, signal_size))
        if verbose:
            labels.append("input")
            tensors.append(dummy)

        self.stlayer = self._create_stlayer(stlayer_config)
        dummy = self.stlayer(dummy)
        if verbose:
            labels.append("After LSTLayer1D")
            tensors.append(dummy)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(num_features=dummy.shape[1])
            dummy = self.batch_norm1(dummy)
            if verbose:
                labels.append("After BatchNorm1D")
                tensors.append(dummy)

        dummy = dummy.reshape(dummy.shape[0], 1, dummy.shape[1], dummy.shape[2])
        self.conv1 = self._create_conv2d(conv1_config, dummy)
        dummy = self.conv1(dummy)
        if verbose:
            labels.append("After Conv2D")
            tensors.append(dummy)

        self.pool1 = self._create_pool2d(pool1_config, dummy)
        dummy = self.pool1(dummy)
        if verbose:
            labels.append("After Pool2D")
            tensors.append(dummy)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(num_features=dummy.shape[1])
            dummy = self.batch_norm2(dummy)
            if verbose:
                labels.append("After BatchNorm2D")
                tensors.append(dummy)

        self.conv2 = self._create_conv2d(conv2_config, dummy)
        dummy = self.conv2(dummy)
        if verbose:
            labels.append("After Conv2D")
            tensors.append(dummy)

        self.pool2 = self._create_pool2d(pool2_config, dummy)
        dummy = self.pool2(dummy)
        if verbose:
            labels.append("After Pool2D")
            tensors.append(dummy)

        if self.batch_norm:
            self.batch_norm3 = nn.BatchNorm2d(num_features=dummy.shape[1])
            dummy = self.batch_norm3(dummy)
            if verbose:
                labels.append("After BatchNorm2D")
                tensors.append(dummy)

        self.flatten = nn.Flatten()
        dummy = self.flatten(dummy)
        self.linear1 = self._create_linear1(linear_config, dummy)
        dummy = self.linear1(dummy)
        if verbose:
            labels.append("After Linear")
            tensors.append(dummy)

        self.linear2 = self._create_linear2(linear_config, dummy)
        dummy = self.linear2(dummy)
        if verbose:
            labels.append("After Linear")
            tensors.append(dummy)

        self.linear3 = self._create_linear3(linear_config, dummy)
        dummy = self.linear3(dummy)
        if verbose:
            labels.append("After Linear")
            tensors.append(dummy)

        if verbose:
            for label, tensor in zip(labels, tensors):
                print(f"{label}, tensor.shape = {tensor.shape}")
    
    def _is_in_range(self, n, a, b):
        return a < n < b
    
    def _validate_inputs(self,
                         stlayer_config,
                         conv1_config,
                         pool1_config,
                         conv2_config,
                         pool2_config,
                         linear_config):
        valid_stlayer_keys = ["scale_min", "scale_max", "num_scales", 
                              "kernel_ratio", "stride_ratio", "kernel_ratio_lp", "stride_ratio_lp"]
        if not isinstance(stlayer_config, dict) or list(stlayer_config.keys()) != valid_stlayer_keys:
            raise ValueError(f"stlayer_config should be a dict object with {valid_stlayer_keys} keys")
        
        valid_conv_keys = ["out_channels", "kernel_ratio", "stride_ratio"]
        valid_kernel_ratio = [0, 1]
        valid_stride_ratio = [0, 1] 
     
        if not isinstance(conv1_config, dict) or list(conv1_config.keys()) != valid_conv_keys:
            raise ValueError(f"conv1_config should be a dict with {valid_conv_keys} keys")
        
        for ratio in conv1_config["kernel_ratio"]:
            if not self._is_in_range(ratio, valid_kernel_ratio[0], valid_kernel_ratio[1]):
                raise ValueError(f"conv1_config[\"kernel_ratio\"] should be in {valid_kernel_ratio}")
        
        for ratio in conv1_config["stride_ratio"]:
            if not self._is_in_range(ratio, valid_stride_ratio[0], valid_stride_ratio[1]):
                raise ValueError(f"conv1_config[\"stride_ratio\"] should be in {valid_stride_ratio}")

        if not isinstance(conv2_config, dict) or list(conv2_config.keys()) != valid_conv_keys:
            raise ValueError(f"conv2_config should be a dict with {valid_conv_keys} keys")
        
        for ratio in conv2_config["kernel_ratio"]:
            if not self._is_in_range(ratio, valid_kernel_ratio[0], valid_kernel_ratio[1]):
                raise ValueError(f"conv2_config[\"kernel_ratio\"] should be in {valid_kernel_ratio}")
        
        for ratio in conv2_config["stride_ratio"]:
            if not self._is_in_range(ratio, valid_stride_ratio[0], valid_stride_ratio[1]):
                raise ValueError(f"conv2_config[\"stride_ratio\"] should be in {valid_stride_ratio}")
        
        valid_pool_keys = ["kernel_ratio", "stride_ratio"]
     
        if not isinstance(pool1_config, dict) or list(pool1_config.keys()) != valid_pool_keys:
            raise ValueError(f"pool1_config should be a dict with {valid_pool_keys} keys")
        
        for ratio in pool1_config["kernel_ratio"]:
            if not self._is_in_range(ratio, valid_kernel_ratio[0], valid_kernel_ratio[1]):
                raise ValueError(f"pool1_config[\"kernel_ratio\"] should be in {valid_kernel_ratio}")
        
        for ratio in pool1_config["stride_ratio"]:
            if not self._is_in_range(ratio, valid_stride_ratio[0], valid_stride_ratio[1]):
                raise ValueError(f"pool1_config[\"stride_ratio\"] should be in {valid_stride_ratio}")
        
        if not isinstance(pool2_config, dict) or list(pool2_config.keys()) != valid_pool_keys:
            raise ValueError(f"pool2_config should be a dict with {valid_pool_keys} keys")
        
        for ratio in pool2_config["kernel_ratio"]:
            if not self._is_in_range(ratio, valid_kernel_ratio[0], valid_kernel_ratio[1]):
                raise ValueError(f"pool2_config[\"kernel_ratio\"] should be in {valid_kernel_ratio}")
        
        for ratio in pool2_config["stride_ratio"]:
            if not self._is_in_range(ratio, valid_stride_ratio[0], valid_stride_ratio[1]):
                raise ValueError(f"pool2_config[\"stride_ratio\"] should be in {valid_stride_ratio}")

        valid_linear_keys = ["hidden1_ratio", "hidden2_ratio", "num_classes"]        
        valid_ratio = [0, 1]
        if not isinstance(linear_config, dict) or list(linear_config.keys()) != valid_linear_keys:
            raise ValueError(f"linear_config should be a dict with {valid_linear_keys} keys")
        
        for ratio in [linear_config["hidden1_ratio"], linear_config["hidden2_ratio"]]:
            if not self._is_in_range(ratio, valid_ratio[0], valid_ratio[1]):
                raise ValueError(f"linear_config[\"hidden_ratio\"] should be in {valid_ratio}")
        
    def _create_stlayer(self, stlayer_config):
        return LSTLayer1D(**stlayer_config)

    def _create_conv2d(self, conv_config, dummy):
        in_channels = dummy.shape[1]
        dummy_shape = dummy.shape[-2:]
        kernel_size = [max(int(ratio * dim), self._MIN_KERNEL_SIZE) for ratio, dim in zip(conv_config["kernel_ratio"], dummy_shape)]
        stride_size = [max(int(ratio * dim), self._MIN_STRIDE_SIZE) for ratio, dim in zip(conv_config["stride_ratio"], kernel_size)]
        layer = nn.Conv2d(in_channels=in_channels,
                          out_channels=conv_config["out_channels"],
                          kernel_size=kernel_size,
                          stride=stride_size,
                          padding=0)
        return layer
    
    def _create_pool2d(self, pool_config, dummy):
        dummy_shape = dummy.shape[-2:]
        kernel_size = [max(int(ratio * dim), self._MIN_KERNEL_SIZE) for ratio, dim in zip(pool_config["kernel_ratio"], dummy_shape)]
        stride_size = [max(int(ratio * dim), self._MIN_STRIDE_SIZE) for ratio, dim in zip(pool_config["stride_ratio"], kernel_size)]
        layer = nn.MaxPool2d(kernel_size=kernel_size,
                             stride=stride_size,
                             padding=0)
        return layer
    
    def _create_linear1(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = max(int(linear_config["hidden1_ratio"] * in_features), linear_config["num_classes"])
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        return layer
    
    def _create_linear2(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = max(int(linear_config["hidden2_ratio"] * in_features), linear_config["num_classes"])
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        return layer
    
    def _create_linear3(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = linear_config["num_classes"]
        layer = nn.Linear(in_features=in_features, out_features=out_features)
        return layer

    def forward(self, x):
        """
        Forward pass of the LSTN network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 1, signal_size).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("input should be a 3D tensor of shape (batch_size, channel_size=1, signal_size)")
        
        x = self.stlayer(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = self.pool1(fcn.relu(x))
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.conv2(x)
        x = self.pool2(fcn.relu(x))
        if self.batch_norm:
            x = self.batch_norm3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(fcn.relu(x))
        logits = self.linear3(fcn.relu(x))

        return logits


class LSTNLF(nn.Module):
    """
    LSTN feature extractor head for Late Fusion (LF) architectures.

    Processes single-channel 1D time-series signals and outputs a flattened feature vector.
    Used as a building block in LSTMADNetLF.

    Architecture:
        LSTLayer1D -> [BatchNorm1D] -> Conv2D -> Pool2D -> [BatchNorm2D] ->
        Conv2D -> Pool2D -> [BatchNorm2D] -> Flatten -> features

    Args:
        signal_size (int): The size of the input signal.
        stlayer_config (dict): Configuration for the LSTLayer1D.
        conv1_config (dict): Configuration for the first Conv2D layer.
        pool1_config (dict): Configuration for the first MaxPool2D layer.
        conv2_config (dict): Configuration for the second Conv2D layer.
        pool2_config (dict): Configuration for the second MaxPool2D layer.
        batch_norm (bool): Whether to include Batch Normalization (default: False).
    
    Input:
        x (torch.Tensor): 3D tensor of shape (batch_size, 1, signal_size).

    Output:
        features (torch.Tensor): 2D tensor of shape (batch_size, feature_size).
    """
    def __init__(self,
                 signal_size,
                 stlayer_config,
                 conv1_config,
                 pool1_config,
                 conv2_config,
                 pool2_config,
                 batch_norm=False):
        
        super().__init__()
        self._MIN_KERNEL_SIZE = 3
        self._MIN_STRIDE_SIZE = 1
        self.batch_norm = batch_norm

        _dummy = torch.randn(size=(1, 1, signal_size))

        self.stlayer = self._create_stlayer(stlayer_config)
        _dummy = self.stlayer(_dummy)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(num_features=_dummy.shape[1])
            _dummy = self.batch_norm1(_dummy)

        _dummy = _dummy.reshape(_dummy.shape[0], 1, _dummy.shape[1], _dummy.shape[2])
        self.conv1 = self._create_conv2d(conv1_config, _dummy)
        _dummy = self.conv1(_dummy)

        self.pool1 = self._create_pool2d(pool1_config, _dummy)
        _dummy = self.pool1(_dummy)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(num_features=_dummy.shape[1])
            _dummy = self.batch_norm2(_dummy)

        self.conv2 = self._create_conv2d(conv2_config, _dummy)
        _dummy = self.conv2(_dummy)

        self.pool2 = self._create_pool2d(pool2_config, _dummy)
        _dummy = self.pool2(_dummy)

        if self.batch_norm:
            self.batch_norm3 = nn.BatchNorm2d(num_features=_dummy.shape[1])
            _dummy = self.batch_norm3(_dummy)

        self.flatten = nn.Flatten()
        _dummy = self.flatten(_dummy)

    def _create_stlayer(self, stlayer_config):
        return LSTLayer1D(**stlayer_config)
    
    def _create_conv2d(self, conv_config, dummy):
        in_channels = dummy.shape[1]
        dummy_shape = dummy.shape[-2:]
        kernel_size = [max(int(ratio * dim), self._MIN_KERNEL_SIZE) for ratio, dim in zip(conv_config["kernel_ratio"], dummy_shape)]
        stride_size = [max(int(ratio * dim), self._MIN_STRIDE_SIZE) for ratio, dim in zip(conv_config["stride_ratio"], kernel_size)]
        layer = nn.Conv2d(in_channels=in_channels,
                          out_channels=conv_config["out_channels"],
                          kernel_size=kernel_size,
                          stride=stride_size,
                          padding=0)
        return layer
    
    def _create_pool2d(self, pool_config, dummy):
        dummy_shape = dummy.shape[-2:]
        kernel_size = [max(int(ratio * dim), self._MIN_KERNEL_SIZE) for ratio, dim in zip(pool_config["kernel_ratio"], dummy_shape)]
        stride_size = [max(int(ratio * dim), self._MIN_STRIDE_SIZE) for ratio, dim in zip(pool_config["stride_ratio"], kernel_size)]
        layer = nn.MaxPool2d(kernel_size=kernel_size,
                             stride=stride_size,
                             padding=0)
        return layer
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("input should be a 3D tensor of shape (batch_size, channel_size=1, signal_size)")
        
        x = self.stlayer(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = self.pool1(fcn.relu(x))
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.conv2(x)
        x = self.pool2(fcn.relu(x))
        if self.batch_norm:
            x = self.batch_norm3(x)
        features = self.flatten(x)

        return features


class LSTMADNetLF(nn.Module):
    """
    LST-MADNet with Late Fusion for multimodal anomaly detection.

    Each channel is processed by an independent LSTNLF head, producing a feature vector.
    Features from all channels are concatenated and passed through fully connected layers.

    Args:
        channels (list): List of channel names (str) from different modalities.
        signal_sizes (dict): Maps channel names to input signal sizes.
        stlayer_configs (dict): LSTLayer1D config for each channel.
        conv1_configs (dict): First Conv2D config for each channel.
        pool1_configs (dict): First Pool2D config for each channel.
        conv2_configs (dict): Second Conv2D config for each channel.
        pool2_configs (dict): Second Pool2D config for each channel.
        batch_norms (dict): Whether BatchNorm is applied for each channel.
        linear_config (dict): Configuration for the fully connected layers.

    Input:
        x_dict (dict): Keys are channel names, values are tensors of shape (batch_size, 1, signal_size).

    Output:
        logits (torch.Tensor): Shape (batch_size, num_classes).
    """
    def __init__(self,
                 channels,
                 signal_sizes,
                 stlayer_configs,
                 conv1_configs,
                 pool1_configs,
                 conv2_configs,
                 pool2_configs,
                 batch_norms,
                 linear_config):
        
        super().__init__()
        self._validate_inputs(channels=channels,
                              signal_sizes=signal_sizes,
                              stlayer_configs=stlayer_configs,
                              conv1_configs=conv1_configs,
                              pool1_configs=pool1_configs,
                              conv2_configs=conv2_configs,
                              pool2_configs=pool2_configs,
                              batch_norms=batch_norms)
        
        self.channels = channels
        self.signal_sizes = signal_sizes
        self.stlayer_configs = stlayer_configs
        self.conv1_configs = conv1_configs
        self.pool1_configs = pool1_configs
        self.conv2_configs = conv2_configs
        self.pool2_configs = pool2_configs
        self.batch_norms = batch_norms
        self.linear_config = linear_config

        _dummy = {channel: torch.randn(1, 1, signal_sizes[channel]) for channel in channels}

        self.heads = nn.ModuleDict({
            channel: LSTNLF(
                signal_size=signal_sizes[channel],
                stlayer_config=stlayer_configs[channel],
                conv1_config=conv1_configs[channel],
                pool1_config=pool1_configs[channel],
                conv2_config=conv2_configs[channel],
                pool2_config=pool2_configs[channel],
                batch_norm=batch_norms[channel]
            )
            for channel in channels
        })

        _latent_dict = {channel: self.heads[channel](_dummy[channel]) for channel in channels}
        _latent = torch.concat([*_latent_dict.values()], dim=1)

        self.linear1 = self._create_linear1(linear_config, _latent)
        _latent = self.linear1(_latent)

        self.linear2 = self._create_linear2(linear_config, _latent)
        _latent = self.linear2(_latent)
        
        self.linear3 = self._create_linear3(linear_config, _latent)
        _latent = self.linear3(_latent)
        
    def _validate_inputs(self, channels, **all_configs):
        for configs_name, configs in all_configs.items():
            if not isinstance(configs, dict):
                raise ValueError(f"{configs_name} should be a dictionary.")
            if set(configs.keys()) != set(channels):
                raise ValueError(f"{configs_name} should be a dictionary with the following keys: {channels}. "
                                 f"Found keys {list(configs.keys())}")
    
    def _create_linear1(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = max(int(linear_config["hidden1_ratio"] * in_features), linear_config["num_classes"])
        return nn.Linear(in_features=in_features, out_features=out_features)
    
    def _create_linear2(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = max(int(linear_config["hidden2_ratio"] * in_features), linear_config["num_classes"])
        return nn.Linear(in_features=in_features, out_features=out_features)
    
    def _create_linear3(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = linear_config["num_classes"]
        return nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x_dict):
        if set(x_dict.keys()) != set(self.channels):
            raise ValueError(f"Input data should be a dictionary with the following keys: {self.channels}. "
                             f"Found keys: {list(x_dict.keys())}")

        latent_dict = {channel: self.heads[channel](x_dict[channel]) for channel in self.channels}
        latent = torch.concat([*latent_dict.values()], dim=1)

        latent = self.linear1(latent)
        latent = self.linear2(fcn.relu(latent))
        logits = self.linear3(fcn.relu(latent))

        return logits


class LSTMADNetDF(nn.Module):
    """
    LST-MADNet with Deep Fusion for multimodal anomaly detection.

    Uses per-channel LSTLayer1D heads to extract time-frequency representations,
    fuses them into a unified 2D representation, then applies shared convolutional
    and fully connected layers.

    Architecture:
        1. Time-Frequency Transformation: Per-channel LSTLayer1D -> Fuse (resize + concat)
        2. Spatial Feature Extraction: Conv2D -> Pool2D -> Conv2D -> Pool2D
        3. Classification: FC -> FC -> FC -> logits

    Args:
        channels (list): List of input channel names.
        signal_sizes (dict): Maps each channel to its input signal size.
        stlayer_configs (dict): LSTLayer1D config for each channel.
        conv1_config (dict): Configuration for the first Conv2D block.
        pool1_config (dict): Configuration for the first Pool2D block.
        conv2_config (dict): Configuration for the second Conv2D block.
        pool2_config (dict): Configuration for the second Pool2D block.
        linear_config (dict): Configuration for fully connected layers.
        batch_norm (bool): Whether to apply BatchNorm2D layers.

    Input:
        x_dict (dict): Keys are channel names, values are tensors of shape (batch_size, 1, signal_size).

    Output:
        logits (torch.Tensor): Shape (batch_size, num_classes).
    """
    def __init__(self,
                 channels,
                 signal_sizes,
                 stlayer_configs,
                 conv1_config,
                 pool1_config,
                 conv2_config,
                 pool2_config,
                 linear_config,
                 batch_norm):
        super().__init__()
        self._validate_inputs(channels=channels,
                              signal_sizes=signal_sizes,
                              stlayer_configs=stlayer_configs)

        self.channels = channels
        self.batch_norm = batch_norm
        self._MIN_KERNEL_SIZE = 3
        self._MIN_STRIDE_SIZE = 1

        dummy_dict = {channel: torch.randn(size=(1, 1, signal_sizes[channel])) for channel in self.channels}

        # Time-Frequency Transformation Stage
        self.heads = nn.ModuleDict({
            channel: LSTLayer1D(**stlayer_configs[channel])
            for channel in self.channels
        })
        dummy_dict = {channel: self.heads[channel](dummy_dict[channel]) for channel in self.channels}
        dummy = self._fuse_data(dummy_dict)
        self.fuse_shape = dummy.shape
        
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(num_features=dummy.shape[1])
            dummy = self.batch_norm1(dummy)

        # Spatial Feature Extraction Stage
        self.conv1 = self._create_conv2d(conv1_config, dummy)
        dummy = self.conv1(dummy)
        
        self.pool1 = self._create_pool2d(pool1_config, dummy)
        dummy = self.pool1(dummy)
        
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(num_features=dummy.shape[1])
            dummy = self.batch_norm2(dummy)
        
        self.conv2 = self._create_conv2d(conv2_config, dummy)
        dummy = self.conv2(dummy)
        
        self.pool2 = self._create_pool2d(pool2_config, dummy)
        dummy = self.pool2(dummy)
        
        if self.batch_norm:
            self.batch_norm3 = nn.BatchNorm2d(num_features=dummy.shape[1])
            dummy = self.batch_norm3(dummy)

        # Classification Mapping Stage
        self.flatten = nn.Flatten()
        dummy = self.flatten(dummy)
        self.linear1 = self._create_linear1(linear_config, dummy)
        dummy = self.linear1(dummy)

        self.linear2 = self._create_linear2(linear_config, dummy)
        dummy = self.linear2(dummy)

        self.linear3 = self._create_linear3(linear_config, dummy)
        dummy = self.linear3(dummy)

    def _validate_inputs(self, channels, **all_configs):
        for configs_name, configs in all_configs.items():
            if not isinstance(configs, dict):
                raise ValueError(f"{configs_name} should be a dictionary.")
            if set(configs.keys()) != set(channels):
                raise ValueError(f"{configs_name} should be a dictionary with the following keys: {channels}. "
                                 f"Found keys {list(configs.keys())}")

    def _fuse_data(self, x_dict):
        """Fuses channel-specific TFRs into a unified 4D tensor via nearest-neighbor interpolation."""
        target_size_1 = max([x_dict[channel].shape[1] for channel in x_dict.keys()])
        target_size_2 = max([x_dict[channel].shape[2] for channel in x_dict.keys()])
        target_sizes = (target_size_1, target_size_2)

        x_dict = {channel: x_dict[channel].unsqueeze(dim=1) for channel in x_dict.keys()}
        x_dict = {channel: fcn.interpolate(input=x_dict[channel],
                                           size=target_sizes,
                                           mode="nearest")
                  for channel in x_dict.keys()}
        
        x = torch.concat(list(x_dict.values()), dim=1)
        return x

    def _create_conv2d(self, conv_config, dummy):
        in_channels = dummy.shape[1]
        dummy_shape = dummy.shape[-2:]
        kernel_size = [max(int(ratio * dim), self._MIN_KERNEL_SIZE) for ratio, dim in zip(conv_config["kernel_ratio"], dummy_shape)]
        stride_size = [max(int(ratio * dim), self._MIN_STRIDE_SIZE) for ratio, dim in zip(conv_config["stride_ratio"], kernel_size)]
        layer = nn.Conv2d(in_channels=in_channels,
                          out_channels=conv_config["out_channels"],
                          kernel_size=kernel_size,
                          stride=stride_size,
                          padding=0)
        return layer
    
    def _create_pool2d(self, pool_config, dummy):
        dummy_shape = dummy.shape[-2:]
        kernel_size = [max(int(ratio * dim), self._MIN_KERNEL_SIZE) for ratio, dim in zip(pool_config["kernel_ratio"], dummy_shape)]
        stride_size = [max(int(ratio * dim), self._MIN_STRIDE_SIZE) for ratio, dim in zip(pool_config["stride_ratio"], kernel_size)]
        layer = nn.MaxPool2d(kernel_size=kernel_size,
                             stride=stride_size,
                             padding=0)
        return layer
    
    def _create_linear1(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = max(int(linear_config["hidden1_ratio"] * in_features), linear_config["num_classes"])
        return nn.Linear(in_features=in_features, out_features=out_features)
    
    def _create_linear2(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = max(int(linear_config["hidden2_ratio"] * in_features), linear_config["num_classes"])
        return nn.Linear(in_features=in_features, out_features=out_features)
    
    def _create_linear3(self, linear_config, dummy):
        in_features = dummy.numel()
        out_features = linear_config["num_classes"]
        return nn.Linear(in_features=in_features, out_features=out_features)
    
    def forward(self, x_dict):
        if set(x_dict.keys()) != set(self.channels):
            raise ValueError(f"Input data should be a dictionary with the following keys: {self.channels}. "
                             f"Found keys: {list(x_dict.keys())}")
        
        # Time-Frequency Transformation Stage
        x_dict = {channel: self.heads[channel](x_dict[channel]) for channel in self.channels}
        x = self._fuse_data(x_dict)
        if self.batch_norm:
            x = self.batch_norm1(x)

        # Spatial Feature Extraction Stage
        x = self.conv1(x)
        x = self.pool1(fcn.relu(x))
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.conv2(x)
        x = self.pool2(fcn.relu(x))
        if self.batch_norm:
            x = self.batch_norm3(x)

        # Classification Mapping Stage
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(fcn.relu(x))
        logits = self.linear3(fcn.relu(x))

        return logits
