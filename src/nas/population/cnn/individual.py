"""
File: individual.py
Author: Jan Kastner
Date: February 29, 2024

Description:
    Individual: The individual of the evolution algorithm.
"""

import os
import random
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

import torch
import random
import torch.nn as nn
import copy
import torch.optim as optim
from encoding.CGP_graph import CGPGraph
from .net import Net
import math


class Individual:
    """
    Represents the individual in the evolution algorithm.

    Attributes:
        __CGP_graph (CGPGraph): The Cartesian Genetic Programming (CGP) representation of the individual.

        calculate_mem (bool): Flag indicating whether to calculate mem.

        calculate_mua (bool): Flag indicating whether to calculate mua.

        __error_rate (float): The error rate of the individual.

        __mem (int): The memory usage of the network which individual represents.

        __mua (int): The number of multiply add operations in the network.

        __wrappers (list): List of wrappers where the index represents the layer type and the value
        represents the wrapper for neural layer.

        __names (list): List of names where the index represents the layer type and the value
        represents the layer name.

        __types (dict): Dictionary storing types corresponding to layer class.

        __layers_backup (torch.nn.ModuleList): Backup of layers.
    """

    def __init__(self, graph_controller, container):
        """
        Initializes a new Individual object.

        Parameters:
            graph_controller: Generates and muatates nodes for CGP.
        """
        self.__CGP_graph = CGPGraph(copy.deepcopy(graph_controller))

        self.calculate_mem = True
        self.calculate_mua = True

        self.__error_rate = 0
        self.__mem = 0
        self.__mua = 0

        self.__names = container.names
        self.__wrappers = container.wrappers
        self.__types = container.types
        self.__layers_backup = nn.ModuleList()

    def mutate(self, mutation_rate):
        """
        Mutates the individual with a given mutation rate.

        Parameters:
            mutation_rate (float): The rate at which mutations occur.

        This method delegates mutation to the CGPGraph object with the specified mutation rate.
        """
        self.__CGP_graph.mutate(mutation_rate)

    def __copy_parameters(self, target_parameters, source_parameters):
        """
        Copies parameters from source (parent) network layer to target (child) network layer.

        Parameters:
            target_parameters (torch.Tensor): The tensor to copy the parameters into.
            source_parameters (torch.Tensor): The tensor containing the parameters to be copied.
        """
        # provides padding (only for biases)
        target_parameters.zero_()

        # target and source sizes doesn't match
        if len(target_parameters.size()) != len(source_parameters.size()):
            return
        # provides xavier_normal_ padding (not for biases)
        elif len(target_parameters.size()) >= 2:
            nn.init.xavier_normal_(target_parameters)

        # clipping of source tensor
        slices = [
            slice(0, min(size1, size2))
            for size1, size2 in zip(target_parameters.size(), source_parameters.size())
        ]

        target_parameters[slices] = source_parameters[slices]

    def __initialize_parameters(self, layers, layers_backup):
        """
        Initializes parameters for layers using parents layers.

        Parameters:
            layers (list): The list of layers to initialize.
            layers_backup (list): The list of parents layers.
        """

        inicialized_layers = 0

        for layer, layer_backup in zip(layers, layers_backup):

            # layer does not occur in either of these two networks
            if layer is None and layer_backup is None:
                continue

            # layer occures only in child network therefore
            # paramter sharing can't be provided
            elif layer_backup is None:

                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                    nn.init.xavier_normal_(
                        layer.weight
                    )  # Initialize weights using Xavier normal
                    nn.init.constant_(layer.bias, 0)  # Initialize bias with zeros
            # layer occures in both child and parent network
            else:
                # elements are layer not nested ModuleLists
                if isinstance(
                    layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)
                ) and isinstance(
                    layer_backup, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)
                ):
                    # parameter sharing can be provided therefore same types of layers
                    # occures in both child and parent layer
                    if type(layer) == type(layer_backup):
                        # Copy parameters from backup layers to current layers
                        self.__copy_parameters(
                            layer.weight.data,
                            layer_backup.weight.data.clone(),
                        )
                        self.__copy_parameters(
                            layer.bias.data,
                            layer_backup.bias.data.clone(),
                        )
                        if layer.weight.size() == layer_backup.weight.size():
                            inicialized_layers += 1
                    # parameter sahring can't be provided therefore layers of child
                    # and parent networks are different types
                    else:
                        # Initialize parameters with Xavier normal if types don't match
                        nn.init.xavier_normal_(layer.weight)
                        nn.init.constant_(layer.bias, 0)
                # recursively initialize parameters for nested ModuleLists
                elif isinstance(layer, nn.ModuleList) and isinstance(
                    layer_backup, nn.ModuleList
                ):
                    inicialized_layers += self.__initialize_parameters(
                        layer, layer_backup
                    )

        return inicialized_layers

    def __num_layers(self, layers):
        count = 0
        for item in layers:
            if isinstance(item, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                count += 1
            elif isinstance(item, nn.ModuleList):
                count += self.__num_layers(item)
        return count

    def evaluate(self, trainloader, testloader, num_epochs):
        """
        Evaluates the individual using training and testing datasets.

        Parameters:
            trainloader: DataLoader for training data.
            testloader: DataLoader for testing data.
            num_epochs (int): Number of epochs for training.
        """
        # Selecting device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Get the first batch from trainloader to determine input shape
        dataiter = iter(trainloader)
        input, _ = next(dataiter)
        input_channels = input.size()[1]  # Number of input channels
        input_size = input.size()[2]  # Size of the image
        self.__net = Net(
            phenotype=self.__CGP_graph.get_active_nodes(),
            wrappers=self.__wrappers,
            channels=input_channels,
            size=input_size,
            output_wrapper=self.__wrappers[self.__types["OUTPUT"][0]],
            cgp_outputs=self.__CGP_graph.get_outputs(),
        )
        # is perfomed when individual does not have parent (initial population)
        if len(list(self.__net.children())) != len(self.__layers_backup):
            self.__layers_backup = nn.ModuleList([None for _ in self.__net.children()])

        self.__net.to(device)
        inicialized_layers = self.__initialize_parameters(
            self.__net.children(), self.__layers_backup
        )
        total_layers = self.__num_layers(self.__net.children())
        inheritance_ratio = inicialized_layers / total_layers
        self.__num_epochs = math.ceil(num_epochs * (1 - inheritance_ratio))

        if self.calculate_mem == True:
            total_params = sum(p.numel() for p in self.__net.parameters())
            self.__mem = total_params

        print(f"Number of parameters: {self.__mem}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.__net.parameters(), lr=0.001, momentum=0.9)
        # Training loop
        for epoch in range(self.__num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self.__net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        # Testing loop
        if self.__num_epochs > 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)

                    outputs = self.__net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            self.__error_rate = 1 - correct / total
            
        print(self.__num_epochs)
        self.__layers_backup = copy.deepcopy(list(self.__net.children()))

        acc = 100 * (1 - self.__error_rate)
        print("Accuracy of the network on the test images: %.2f %%" % acc)

    @property
    def fitness_value(self):
        """
        Property method to retrieve the fitness value of the individual.

        Returns:
            list: List containing the error rate, memory usage (if enabled) and memory update amount
            (if enabled).
        """
        fitness = []
        fitness.append(self.__error_rate)
        if self.calculate_mem == True:
            fitness.append(self.__mem)
        if self.calculate_mua == True:
            fitness.append(self.__mua)
        return fitness

    @property
    def graph_info(self):
        """
        Property method to retrieve detailed information about the individual.

        Returns:
            dict: Dictionary containing genotype, phenotype, fitness value, outputs, shape and names.
        """
        shape = self.__CGP_graph.shape
        genotype = [
            [node.get_type(), node.get_input(1), node.get_input(2)]
            for node in self.__CGP_graph.get_nodes()
        ]
        phenotype = []
        for node in self.__CGP_graph.get_active_nodes():
            if node is not None:
                phenotype.append(
                    [node.get_type(), node.get_input(1), node.get_input(2)]
                )
            else:
                phenotype.append(None)

        return {
            "genotype": genotype,
            "phenotype": phenotype,
            "outputs": self.__CGP_graph.get_outputs(),
            "shape": shape,
            "names": self.__names,
        }

    @property
    def net(self):
        return self.__net
