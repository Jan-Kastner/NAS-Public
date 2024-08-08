"""
File: individual.py
Author: Jan Kastner
Date: February 29, 2024

Description:
    Individual: An individual of an evolution algorithm.
    _Net:       A network represented by the individual.
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

import torch
import torch.nn as nn
import torch.nn.functional as F
from population.wrappers import Cat as catw, Add as addw, Conv2d as conv


class Net(nn.Module):

    CONVOLUTIONAL_LAYER_IDX = 0
    CONVOLUTIONAL_NORMALIZATION_IDX = 1

    MERGING_LAYER_I_IDX = 0
    MERGING_LAYER_II_IDX = 1
    MERGING_NORMALIZATION_IDX = 2
    """
    The network represented by the individual.

    Attributes:
        layers (nn.ModuleList): List to store and properly register layers.

        __phenotype (list): active cgp nodes.

        __cgp_outputs (list): outputs of cgp.

        __merging_strategies (list): strategies used for layer merging.
    """

    def __init__(
        self, phenotype, wrappers, size, channels, output_wrapper, cgp_outputs
    ):
        """
        Initializes a new CNN object and calculates the sizes of output tensors of individual layers.

        Parameters:
            phenotype (list): The list of values of active CGPnodes.

            wrappers (list): The list of wrappers (used for dynamic CNN layers).

            size (int): Spatial dimensions of input tensor.

            channels (int): The number of channels of input.

            output_wrapper (Linear): The wrapper for a linear layer which is used as an output layer of the
            network.

            cgp_outputs (list): The indexies of output nodes in cgp (in the network it is an index of the
            output layer).
        """
        super(Net, self).__init__()
        # Append None to layers list which represents padding
        # (in cgp nodes are numbered from (len(inputs)-1)+1 and None represents the input)
        self.__phenotype = [None] + phenotype
        self.__merging_strategies = []
        self.__wrappers = wrappers
        self.__output_wrapper = output_wrapper
        self.__cgp_outputs = cgp_outputs
        self.layers = nn.ModuleList()
        # outputs of network layers given by input
        outputs = []
        # input tensor informations
        outputs.append(
            {"size": size, "channels": channels, "features": size * size * channels}
        )
        # Append None to layers list which represents padding
        # (in cgp nodes are numbered from (len(inputs)-1)+1 and None represents the input)
        self.layers.append(None)

        for node in phenotype:
            # outputs of unused layers (inactive nodes in CGP graph)
            if node is None:
                outputs.append(
                    {"size": 0, "channels": 0, "features": 0}
                )  # Add placeholder output
                # non-existing layer is represented by None
                self.layers.append(None)
                continue
            # Get wrapper for current node
            wrapper = self.__wrappers[node.get_type()]

            # Check if wrapper wrapps merging layer
            if isinstance(wrapper, (addw, catw)):
                # inputs to be merged
                input1 = outputs[node.get_input(1)]
                input2 = outputs[node.get_input(2)]
                # layers used for transforming tensors to be compatible for merging
                transformers = wrapper.get_transformers(input1, input2)
                # normalization for convolutional layer
                if isinstance(
                    transformers[self.MERGING_LAYER_I_IDX], nn.ConvTranspose2d
                ):
                    normalization = nn.BatchNorm2d(
                        transformers[self.MERGING_LAYER_I_IDX].out_channels
                    )
                elif isinstance(
                    transformers[self.MERGING_LAYER_II_IDX], nn.ConvTranspose2d
                ):
                    normalization = nn.BatchNorm2d(
                        transformers[self.MERGING_LAYER_II_IDX].out_channels
                    )
                else:
                    normalization = None
                # append normalization to transformers
                transformers += [normalization]
                self.layers.append(nn.ModuleList(transformers))
                # store merging strategy either summation (add) or concatenation (cat)
                self.__merging_strategies.append(wrapper.get_layer())
            # wrapper which wrapps convolutional, linear or pooling layer
            else:
                input1 = outputs[node.get_input(1)]
                # getting layer based on input [input tensor affects layer parameters]
                layer = wrapper.get_layer(input1)
                if isinstance(wrapper, conv):
                    normalization = nn.BatchNorm2d(layer.out_channels)
                    self.layers.append(nn.ModuleList([layer, normalization]))
                else:
                    self.layers.append(layer)

            # append output tensor to output list
            outputs.append(wrapper.output_size)

        # output
        input = outputs[
            cgp_outputs[0]
        ]  # in this representation cgp has only one output
        layer = self.__output_wrapper.get_layer(input)
        self.layers.append(layer)
        outputs.append(self.__output_wrapper.output_size)

    def process_transformers(self, transformer, input, normalization=None):
        """
        Process transformers (if any) for a given layer and input data.

        Parameters:
            layer (nn.Module or None): The transformer to be processed.

            input_data (torch.Tensor): The input data to be processed.

        Returns:
            torch.Tensor: Processed output data.
        """
        # transformation is not need
        if transformer is None:
            return input.clone()
        # convoluitional transformation
        elif isinstance(transformer, nn.ConvTranspose2d):
            # convolutional transformation with normalization
            if isinstance(normalization, nn.BatchNorm2d):
                return F.relu(normalization(transformer(input)))
            # convolutional transformation
            else:
                return F.relu(transformer(input))
        # pooling transformation
        else:
            return transformer(input)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ms_idx = 0  # Index for merging strategies
        outputs = [
            None for _ in range(len(self.layers))
        ]  # Initialize outputs list with None

        outputs[0] = x  # Set input tensor as output of None layer

        for idx, _ in enumerate(self.layers):
            if self.layers[idx] is None:  # Skip if layer is None
                continue
            # Check if layer is ModuleList
            elif isinstance(self.layers[idx], nn.ModuleList):
                # Convolutional layer with batch normalization (conv + batch)
                if isinstance(
                    self.layers[idx][self.CONVOLUTIONAL_NORMALIZATION_IDX],
                    nn.BatchNorm2d,
                ):
                    i_idx = self.__phenotype[idx].get_input(1)
                    outputs[idx] = F.relu(
                        self.layers[idx][self.CONVOLUTIONAL_NORMALIZATION_IDX](
                            self.layers[idx][self.CONVOLUTIONAL_LAYER_IDX](
                                outputs[i_idx]
                            )
                        )
                    )
                # merging layer (conv/pool/None + conv/pool/None + norm/None)
                else:
                    t1 = self.process_transformers(
                        transformer=self.layers[idx][self.MERGING_LAYER_I_IDX],
                        input=outputs[self.__phenotype[idx].get_input(1)],
                        normalization=self.layers[idx][self.MERGING_NORMALIZATION_IDX],
                    )
                    t2 = self.process_transformers(
                        transformer=self.layers[idx][self.MERGING_LAYER_II_IDX],
                        input=outputs[self.__phenotype[idx].get_input(2)],
                        normalization=self.layers[idx][self.MERGING_NORMALIZATION_IDX],
                    )

                    merging_strategy = self.__merging_strategies[
                        ms_idx
                    ]  # Get merging strategy
                    ms_idx += 1
                    if merging_strategy == torch.cat:
                        outputs[idx] = merging_strategy((t1, t2), dim=1)
                    else:
                        outputs[idx] = merging_strategy(t1, t2)
            else:
                # Output layer
                if idx == len(self.__phenotype):
                    i_idx = self.__cgp_outputs[
                        0
                    ]  # in this representation cgp has only one output
                else:
                    i_idx = self.__phenotype[idx].get_input(1)
                # Convolutional layer
                if isinstance(self.layers[idx], nn.Conv2d):
                    outputs[idx] = F.relu(self.layers[idx](outputs[i_idx]))
                # Linear layer
                elif isinstance(self.layers[idx], nn.Linear):
                    flatten_input = torch.flatten(
                        outputs[i_idx], 1
                    )  # Flatten all dimensions except batch
                    outputs[idx] = F.relu(self.layers[idx](flatten_input))
                # Pooling layer
                else:
                    outputs[idx] = self.layers[idx](outputs[i_idx])

        return outputs[-1]  # Return output
