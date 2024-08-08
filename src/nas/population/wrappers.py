"""
File: wrappers.py
Author: Jan Kastner
Date: February 29, 2024

Description:    
    _Transformers: Provides neural layers which transformers inputs to be compatible for 
    merging.
    
    Add: Represents a wrapper for a neural layer which provides merging based on 
    summation of input tensors.
    
    Con: Represents a wrapper for a neural layer which provides merging based on 
    concatenation of input tensors.
    
    Conv2d: Represents a wrapper for convolutional neural layer.
    
    Linear: Represents a wrapper for fully connected neural layer.
    
    AvgPool2d: represents a wrapper for avarage pooling layer.
    
    MaxPool2d: represents a wrapper for max pooling layer.
    
    _PoolSizeCalculator: Provides shape of the output tensor after applying the pooling layer.
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

import torch
import torch.nn as nn
import copy
import math


class _Transformers:
    """
    Represents layers which must be applied to input tensors to make them compatible.

    Attributes:
            __size (int): Spatial dimnesions of the output tensor.

            __out_channels (int): Number of channels in the output tensor.
    """

    def __init__(self):
        """
        Initializes new _Transformers object.
        """
        self.__size = 0
        self.__out_channels = 0

    def get_transformers(self, input1, input2, same_channels=True):
        """
        This function determines whether the sizes of input tensors match. If not, it creates
        neural layers to adjust input tensors shape to ensure their compatibility for merging.

        Parameters:
            input1 (dict): Informations about the first input tensor  containing 'size', 'channels' and
            'features'.

            input2 (dict): Informations about the second input tensor  containing 'size', 'channels' and
            'features'.

        Returns:
            list: List containing two neural layers (Pool2d and Conv2d) to adjust tensor sizes for
            concatenation.
                Returns [None, None] if the sizes of the input tensors already match.
        """
        if (
            input1["channels"] == input2["channels"]
            and input1["size"] == input2["size"]
        ):
            self.__size = input1["size"]
            if same_channels:
                self.__out_channels = input1["channels"]
            else:
                self.__out_channels = input1["channels"] * 2
            return [None, None]

        i1_is_bigger = input1["size"] > input2["size"]

        if i1_is_bigger:
            bigger = copy.deepcopy(input1)
            smaller = copy.deepcopy(input2)
        else:
            smaller = copy.deepcopy(input1)
            bigger = copy.deepcopy(input2)

        # Calculate kernel size for MaxPool2d layer
        # H_out = ⌊((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1⌋
        # stride = kernel_size, dilation = 1, padding = 0 ---> kernel_size = H_in / H_out

        kernel_size = float(bigger["size"]) / float(smaller["size"])
        kernel_size = math.floor(kernel_size)
        bigger["size"] = ((bigger["size"] - kernel_size) / kernel_size) + 1
        bigger["size"] = math.floor(bigger["size"])
        # Create MaxPool2d layer if kernel_size > 1
        max_pool_layer = (
            nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
            if kernel_size > 1
            else None
        )

        # Calculate kernel size for Conv2d layer
        # H_out = ⌊((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1⌋
        # stride = 1, dilation = 1, padding = kernel_size - 1 ---> kernel_size = H_out - H_in + 1
        stride = bigger["size"] // smaller["size"]
        kernel_size = bigger["size"] - ((smaller["size"] - 1) * stride)
        if same_channels:
            in_channels = smaller["channels"]
            out_channels = bigger["channels"]
            self.__out_channels = bigger["channels"]
        else:
            in_channels = smaller["channels"]
            out_channels = smaller["channels"]
            self.__out_channels = smaller["channels"] + bigger["channels"]
        # Create Conv2d layer
        upsampling = nn.ConvTranspose2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            out_channels=out_channels,
            in_channels=in_channels,
        )

        self.__size = bigger["size"]

        return (
            [max_pool_layer, upsampling]
            if i1_is_bigger
            else [upsampling, max_pool_layer]
        )

    @property
    def output_size(self):
        """
        Computes size of the merging layer output.

        Returns:
            dict: Information about the output tensor containing 'size', 'channels' and 'features'.
        """
        out_features = self.__size * self.__size * self.__out_channels
        return self.__size, self.__out_channels, out_features


class Add:
    """
    Represents a layer which provides summation of two layers. Its type of concatenation
    where are elements in tensors element wise summated.

    Attributes:
            __size            (int): Spatial dimnesions of the output tensor.

            __out_channels    (int): Number of channels in the output tensor.
    """

    def __init__(self):
        """
        Initializes new Add object.
        """
        self.__transformers = _Transformers()

    def get_transformers(self, input1, input2):
        """
        Retrieves transformers from _Transformers instance for given inputs.

        Parameters:
            input1 (dict): Informations about the first input tensor  containing 'size', 'channels' and
            'features'.

            input2 (dict): Informations about the second input tensor  containing 'size', 'channels' and
              'features'.

        Returns:
            Transformers: The transformers for given inputs.
        """
        return self.__transformers.get_transformers(input1, input2)

    def get_layer(self):
        """
        Merges two tensors according to the specified merging strategy.

        Parameters:
            a (tensor): First tensor to be merged.
            b (tensor): Second tensor to be merged.

        Returns:
            Tensor: Merged tensor according to the specified strategy.
        """
        return torch.add

    @property
    def output_size(self):
        size, out_channels, out_features = self.__transformers.output_size
        return {"size": size, "channels": out_channels, "features": out_features}


class Cat:
    """
    Represents a layer which provides concatenation of two layers. Its type of merging
    where two tensors are concatenated. Batch size is preserved.

    Attributes:
            __size            (int): Spatial dimnesions of the output tensor.
            __out_channels    (int): Number of channels in the output tensor.
    """

    def __init__(self):
        """
        Initializes new Add object.
        """
        self.__transformers = _Transformers()

    def get_transformers(self, input1, input2):
        """
        Retrieves transformers from _Transformers instance for given inputs.

        Parameters:
            input1 (dict): Informations about the first input tensor  containing 'size', 'channels' and
            'features'.

            input2 (dict): Informations about the second input tensor  containing 'size', 'channels' and
            'features'.

        Returns:
            Transformers: The transformers for given inputs.
        """
        return self.__transformers.get_transformers(input1, input2, same_channels=False)

    def get_layer(self):
        """
        Merges two tensors according to the specified merging strategy.

        Parameters:
            a (tensor): First tensor to be merged.
            b (tensor): Second tensor to be merged.

        Returns:
            Tensor: Merged tensor according to the specified strategy.
        """
        return torch.cat

    @property
    def output_size(self):
        size, out_channels, out_features = self.__transformers.output_size
        return {"size": size, "channels": out_channels, "features": out_features}


class Conv2d:
    """
    Stores parameters for a convolutional layer. After providing an input tensor, the neural network
    layer is created. An instance of this class is used for preparing the layer before network creation.

    Attributes:
            __kernel_size (int): Size of the convolutional kernel.

            __stride (int): Stride of the convolution.

            __padding (int): Zero-padding.

            __out_channels (int): Number of channels produced by the convolution.

            __self.__input (dict): Shape of input tensor. Contains informations about size, channels, and
            features.
    """

    def __init__(self, kernel_size, stride, padding, out_channels):
        """
        Initializes new ConvWrapper object.
        """
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__out_channels = out_channels
        self.__input = {"size": 0, "channels": 0, "features": 0}

    def get_layer(self, input):
        """
        Constructs the convolutional layer based on the input tensor.

        Parameters:
            input (dict): The shape of the input tensor. Contains informations about 'size', 'channels', and
            'features'.

        Returns:
            nn.Conv2d: The convolutional layer with specified parameters.
        """
        self.__input = input
        return nn.Conv2d(
            kernel_size=self.__kernel_size,
            stride=self.__stride,
            padding=self.__padding,
            out_channels=self.__out_channels,
            in_channels=input["channels"],
        )

    @property
    def output_size(self):
        """
        Computes the size of the convolutional layer output.

        Returns:
            dict: Informations about the output tensor. Contains: 'size', 'channels' and 'features'.
        """
        # H_out = ⌊((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1⌋
        dilation = 1
        size = math.floor(
            (
                (
                    self.__input["size"]
                    + 2 * self.__padding
                    - dilation * (self.__kernel_size - 1)
                    - 1
                )
                / self.__stride
            )
            + 1
        )
        out_features = size * size * self.__out_channels
        return {"size": size, "channels": self.__out_channels, "features": out_features}


class Linear:
    """
    Stores parameters for a linear layer. After providing an input tensor, the neural network layer is
    created. An instance of this class is used for preparing the layer before network creation.

    Attributes:
            __out_features (int): Number of features produced by layer.
            __out_features (int): Number of features expected by layer.
    """

    def __init__(self, out_features):
        """
        Initializes new LinearWrapper object.
        """
        self.__out_features = out_features

    def get_layer(self, input):
        """
        Constructs a linear layer based on input tensor.

        Parameters:
            input (dict): Shape of input tensor. Contains informations about size, channels, and features.

        Returns:
            nn.Linear: Linear layer with the specified input features.
        """
        return nn.Linear(
            in_features=input["features"], out_features=self.__out_features
        )

    @property
    def output_size(self):
        """
        Computes size of the linear layer output.

        Returns:
            dict: Information about the output tensor. Contains: 'size', 'channels' and 'features'.
        """
        return {"size": 0, "channels": 0, "features": self.__out_features}


class _PoolSizeCalculator:
    @staticmethod
    def calculate(H_in, padding, kernel_size, stride, channels_in):
        """
        Computes size of the pooling layer output.

        Returns:
            dict: Informations about the output tensor. Contains: 'size', 'channels' and 'features'.
        """
        dilation = 1
        size = math.floor(
            ((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        out_features = size * size * channels_in
        return {"size": size, "channels": channels_in, "features": out_features}


class MaxPool2d:
    """
    Stores settings of a pooling layer. An instance of this class is used for preparing the layer before
    network creation.

    Attributes:
            __kernel_size (int): Size of the convolutional kernel.
            __stride (int): Stride of the convolution.
            __padding (int): Zero-padding.
    """

    def __init__(self, kernel_size, stride, padding):
        """
        Initializes new PoolingWrapper object.
        """
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

    def get_layer(self, input):
        """
        Constructs a pooling layer.

        Parameters:
            input (dict): Shape of the input tensor. Contains informations about size, channels, and
            features.

        Returns:
            nn.AvgPool2d: specified pooling layer.
        """
        self.__input = input
        return nn.MaxPool2d(
            kernel_size=self.__kernel_size, stride=self.__stride, padding=self.__padding
        )

    @property
    def output_size(self):
        return _PoolSizeCalculator.calculate(
            self.__input["size"],
            self.__padding,
            self.__kernel_size,
            self.__stride,
            self.__input["channels"],
        )


class AvgPool2d:
    """
    Stores settings of a avarage pooling layer. An instance of this class is used for preparing the layer
    before network creation.

    Attributes:
            __kernel_size (int): Size of the convolutional kernel.
            __stride (int): Stride of the convolution.
            __padding (int): Zero-padding.
    """

    def __init__(self, kernel_size, stride, padding):
        """
        Initializes new PoolingWrapper object.
        """
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

    def get_layer(self, input):
        """
        Constructs a pooling layer.

        Parameters:
            input (dict): Shape of the input tensor. Contains informations about size, channels, and
            features.

        Returns:
            nn.AvgPool2d: specified pooling layer.
        """
        self.__input = input
        return nn.AvgPool2d(
            kernel_size=self.__kernel_size, stride=self.__stride, padding=self.__padding
        )

    @property
    def output_size(self):
        return _PoolSizeCalculator.calculate(
            self.__input["size"],
            self.__padding,
            self.__kernel_size,
            self.__stride,
            self.__input["channels"],
        )
