"""
File: container.py
Author: Jan Kastner
Date: February 29, 2024

Description:    
    Container:  Represents set of layers which can occure in network which is represented by 
    individual.
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

"""
File: example.py
Author: Your Name
Date: February 29, 2024

Description: This file contains example code to demonstrate proper header formatting in Python files.
"""

from itertools import product
from population.wrappers import (
    Add as addw,
    Cat as catw,
    Conv2d as conw,
    AvgPool2d as avgw,
    MaxPool2d as maxw,
    Linear as linw,
)


class Container:
    """
    Stores informations about layers from which neural network can be composed.

    Attributes:
        __wrappers (list): List of wrappers where the index represents the layer type and the value
        represents the wrapper for neural layer.

        __names (list): List of names where the index represents the layer type and the value
        represents the layer name.

        __types (dict): Dictionary storing types corresponding to layer class. It has folowing structure:
                    self._template_types = {
                        'CL': [],
                        'PL': [],
                        'FC': [],
                        'ML': [],
                        'OUTPUT': []}
    """

    def __init__(self):
        """
        Initializes a new Container object.
        """
        self.__wrappers = []
        self.__names = []
        self.__types = {"CL": [], "PL": [], "FC": [], "ML": [], "OUTPUT": []}
        self.__type = 0

    def __to_list(self, data):
        """
        Converts a non-list data into a single-element list or returns the input list as is.

        Parameters:
            data (list or non-list): The data to be converted if not a list.

        Returns:
            list: Converted data.
        """
        return data if isinstance(data, list) else [data]

    def __positive_integers(self, lst):
        """
        Checks whether the input list contains only positive integers.

        Parameters:
            lst (list): The list to be filtered.

        Returns:
            bool: True if input list contains only positive integers, False otherwise.
        """
        return all(x is None or (isinstance(x, int) and x > 0) for x in lst)

    def add(
        self,
        layer_class=None,
        layer_subclass=None,
        kernel_sizes=None,
        strides=None,
        out_channels=None,
        out_features=None,
    ):
        """
        Creates new layers and stores informations into '__wrappers', '__names', '__types'.

        Parameters:
            layer_class (str, optional): The class of the layer.
                Must be one of the following:
                - 'CL' for convolutional layer.
                - 'PL' for pooling layer.
                - 'FC' for fully connected layer.
                - 'ML' for merging layer.
                - 'OUTPUT' for output layer.
                Notes:  Only one output layer is supported.
                        If several of them are entered, the last one entered is taken into account.

            layer_subclass (str, optional): The subclass of the layer.
                Must be one of the following:
                - 'CON' for concatenation layer.
                - 'ADD' for ADD layer.
                Notes: It is only considered if 'layer_class' value is 'ML'.

            kernel_sizes (list, , optional): The kernel sizes used in layers.
                Only square kernels are supported e.g. for kernel_sizes = 2 the kernel size will be 2x2.
                Notes: It is only considered if layer_class value is either 'CL' or 'PL'.

            strides (list, , optional): The strides used in layers.
                Stride cannot be larger than the kernel size.
                Notes: It is only considered if layer_class value is either 'CL' or 'PL'.

            out_channels (list, , optional): Number of channels produced by layers.
                Notes: It is only considered if layer_class value is either 'CL' or 'PL'.

            out_features (list, , optional): Number of channels produced by layers..
                Notes: It is only considered if layer_class value is 'FC'.

        Raises:
            ValueError: If any of the following conditions are met:
            - kernel_sizes, strides, or out_channels are not integers or lists.
            - kernel_sizes, strides, or out_channels contain non-positive elements.
            - the necessary parameters to create the layer are not provided
            - if layer_class or layer_subclass (when needed) are incorrect

        Notes: layer type is derived from the order in which the layer was added.
        """

        params = [
            self.__to_list(kernel_sizes),
            self.__to_list(strides),
            self.__to_list(out_channels),
            self.__to_list(out_features),
        ]

        # check for positive integers
        for param in params:
            if not self.__positive_integers(param):
                raise ValueError(
                    "All elements of '{}' must be positive integers.".format(param)
                )

        kernel_sizes, strides, out_channels, out_features = params

        # Iterate over the cartesian product of kernel sizes, strides, output channels, and output
        # features.
        for params in product(kernel_sizes, strides, out_channels, out_features):
            # unpack parameters
            k_s, s, o_ch, o_f = params

            # convolutional layers
            if (
                layer_class == "CL"
                and k_s is not None
                and s is not None
                and o_ch is not None
            ):
                self.__names.append(
                    str(layer_class)
                    + "\n"
                    + str(k_s)
                    + "X"
                    + str(k_s)
                    + "S"
                    + str(s)
                    + "CH"
                    + str(o_ch)
                )
                self.__wrappers.append(
                    conw(
                        kernel_size=k_s,
                        stride=s,
                        padding=int(k_s / 2),
                        out_channels=o_ch,
                    )
                )

            # pooling layers
            elif layer_class == "PL" and k_s is not None and s is not None:
                self.__names.append(
                    str(layer_subclass)
                    + "\n"
                    + str(k_s)
                    + "X"
                    + str(k_s)
                    + "S"
                    + str(s)
                )
                if layer_subclass == "AVG":
                    self.__wrappers.append(
                        avgw(kernel_size=k_s, stride=s, padding=int(k_s / 2))
                    )
                elif layer_subclass == "MAX":
                    self.__wrappers.append(
                        maxw(kernel_size=k_s, stride=s, padding=int(k_s / 2))
                    )
                else:
                    raise ValueError(
                        "Invalid 'layer_subclass': '{}'. Must be 'MAX' or 'AVG'.".format(
                            layer_subclass
                        )
                    )
            # fully connected layers
            elif layer_class == "FC" and o_f is not None:
                self.__names.append(str(layer_class) + "\n" + str(o_f))
                self.__wrappers.append(linw(out_features=o_f))

            # merging layers
            elif layer_class == "ML":
                if layer_subclass == "CON":
                    self.__names.append("CON")
                    self.__wrappers.append(catw())
                elif layer_subclass == "ADD":
                    self.__names.append("ADD")
                    self.__wrappers.append(addw())
                else:
                    raise ValueError(
                        "Invalid 'layer_subclass': '{}'. Must be 'CON' or 'ADD'.".format(
                            layer_subclass
                        )
                    )
            # output layer
            elif layer_class == "OUTPUT" and o_f is not None:
                self.__names.append("OUTPUT")
                self.__wrappers.append(linw(out_features=o_f))

            else:
                raise ValueError(
                    "Invalid 'layer_class' or not all parameters were provided to create the layer."
                )

            # there can be only one output layer
            if layer_class == "OUTPUT":
                self.__types[layer_class] = []
                self.__types[layer_class].append(self.__type)
            # non-output layers
            else:
                self.__types[layer_class].append(self.__type)
            self.__type += 1

    @property
    def types(self):
        return self.__types

    @property
    def wrappers(self):
        return self.__wrappers

    @property
    def names(self):
        return self.__names

    @property
    def specified_conv_layers(self):
        return len(self.__types['CL']) > 0
    
    @property
    def specified_pooling_layers(self):
        return len(self.__types['PL']) > 0
    
    @property
    def specified_merging_layers(self):
        return len(self.__types['ML']) > 0
    
    @property
    def specified_fully_connected_layers(self):
        return len(self.__types['FC']) > 0
    
    @property
    def specified_output_layer(self):
        return len(self.__types['OUTPUT']) > 0
