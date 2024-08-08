"""
File: generator.py
Author: Jan Kastner
Date: February 29, 2024

Description:    
    Template:  Represents node which is used by an individual to create the genotype.
"""

import queue
import random


class CGPGraph:
    def __init__(self, graph_controller):
        """
        Initializes a new CGPGraph object.

        Parameters:
            seed (Seed): The seed used for initialization. Used for additinal restrictions on functions and edges.
        """
        self.__graph_controller = graph_controller
        # inputs of cgp graph
        self.__inputs = graph_controller.inputs()
        # nodes of cgp graph
        self.__nodes = graph_controller.nodes()
        # outputs of cgp graph
        self.__outputs = graph_controller.outputs()

        # Check types and correctness of inputs, nodes, and outputs
        if not (
            isinstance(self.__inputs, list)
            and isinstance(self.__nodes, list)
            and isinstance(self.__outputs, list)
        ):
            return None
        if not (all(item == index for index, item in enumerate(self.__inputs))):
            return None
        elif not (all(isinstance(node, Node) for node in self.__nodes)):
            return None

        # Initialize attributes
        self.__number_of_nodes = len(self.__nodes)
        self.__number_of_inputs = len(self.__inputs)
        self.__number_of_outputs = len(self.__outputs)
        self.__alleles_in_node = self.__nodes[0].number_of_alleles
    def mutate(self, mutation_rate):
        """
        Mutates the genotype.

        Parameters:
            mutation_rate (float): The mutation rate.

        Notes:
            Mutates the nodes and outputs of the CGPGraph based on the mutation rate.
        """
        number_of_alleles = (
            self.__number_of_nodes * self.__alleles_in_node + self.__number_of_outputs
        )  # Total number of alleles

        alleles_to_mutate = number_of_alleles * mutation_rate
        alleles_to_mutate = max(
            1, int(alleles_to_mutate)
        )  # Number of alleles to mutate

        for _ in range(alleles_to_mutate):
            allele = random.randint(0, number_of_alleles - 1)  # Select a random allele
            # Mutate node
            if (allele // self.__alleles_in_node) < self.__number_of_nodes:
                idx = allele // self.__alleles_in_node
                n_position = idx + self.__number_of_inputs
                a_position = allele - idx * self.__alleles_in_node
                node = self.__nodes[idx]
                self.__nodes[idx] = self.__graph_controller.mutate_node(
                    n_position=n_position, a_position=a_position, node=node
                )
            # Mutate output
            else:
                idx = allele - (self.__number_of_nodes * self.__alleles_in_node)
                output = self.__outputs[idx]
                self.__outputs[idx] = self.__graph_controller.mutate_output()

    def get_nodes(self):
        """
        Returns the nodes of the CGPGraph.

        Returns:
            list: List of nodes.
        """
        return self.__nodes

    def get_inputs(self):
        """
        Returns the inputs of the CGPGraph.

        Returns:
            list: List of inputs.
        """
        return self.__inputs

    def get_outputs(self):
        """
        Returns the outputs of the CGPGraph.

        Returns:
            list: List of outputs.
        """
        return self.__outputs

    def get_active_nodes(self):
        """
        Extracts phenotype from genotype.

        Returns:
            list: Values of active CGP nodes (phenotype).

        Notes:
            The list is the same size as the genotype.
            At the indexes where the inactive node is located, it is assigned the value None.
        """
        # Initial phenotype
        active_nodes = [None] * len(self.__nodes)
        # Queue to hold unprocessed nodes
        nodes_to_process = queue.Queue()
        # Initialize with the last node
        for output in self.__outputs:
            idx = output - self.__number_of_inputs
            nodes_to_process.put(self.__nodes[idx])
            active_nodes[idx] = self.__nodes[idx]

        # Process nodes
        while nodes_to_process.qsize() >= 1:
            for n_position in nodes_to_process.get().get_inputs():
                if n_position is not None:
                    idx = n_position - self.__number_of_inputs
                    if idx < 0:
                        continue
                    nodes_to_process.put(self.__nodes[idx])
                    active_nodes[idx] = self.__nodes[idx]
        return active_nodes

    @property
    def shape(self):
        """
        Returns the shape of the CGP graph.

        Returns:
            tuple: Shape of the CGP graph.
        """
        return self.__graph_controller.shape


class Node:
    """
    Represents node of CGP graph.

    Attributes:
        self.__alleles          (list): The list where alleles are stored.
        self.__type_position    (int): The position of type in alleles.
        self.__number_of_inputs (Generator): The number of edges entering the node.
    """

    def __init__(self, number_of_inputs):
        """
        Initializes a new CGPNode object.

        Parameters:
            number_of_inputs (int): Number of inputs for the CGP node.
        """
        if not isinstance(number_of_inputs, int):
            return False  # If number_of_inputs is not an integer, return False

        self.__number_of_inputs = number_of_inputs
        self.__type_position = 0
        # + 1 is space for type
        self.__alleles = [
            None for _ in range(number_of_inputs + 1)
        ]  # Initialize alleles list

    def set_type(self, value):
        """
        Sets the type of the CGPNode.

        Parameters:
            value (int): The type value to be set.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not isinstance(value, int):
            return False  # If value is not an integer, return False

        self.__alleles[self.__type_position] = value
        return True

    def set_input(self, idx, value):
        """
        Sets the input value of the CGPNode.

        Parameters:
            idx (int): Index of the input to be set.
            value (int): The value to be set as input.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not (
            isinstance(idx, int)
            and isinstance(value, int)
            and 1 <= idx <= self.__number_of_inputs
        ):
            return False  # If idx or value are not integers or idx is out of bounds, return False

        self.__alleles[idx] = value
        return True

    def get_type(self):
        """
        Returns the type of the function represented by CGPNode.

        Returns:
            int: The type of the function represented by CGPNode.
        """
        return self.__alleles[0]

    def get_input(self, idx):
        """
        Returns index of node connected to input on index 'idx'.

        Parameters:
            idx (int): The index of input.

        Returns:
            int/None: The index of node connected to input on index 'idx'. Returns None if index is invalid.
        """
        if not (isinstance(idx, int) and 1 <= idx <= self.__number_of_inputs):
            return None  # If idx is not an integer or out of bounds, return None
        else:
            return self.__alleles[idx]

    def get_inputs(self):
        """
        Returns all indexies of nodes connected to current CGPNode.

        Returns:
            list: List of indexies of nodes connected to current CGPNode.
        """
        return self.__alleles[1:]

    @property
    def number_of_alleles(self):
        """
        Returns the number of alleles (including the type) of the CGPNode.

        Returns:
            int: Number of alleles of the CGPNode.
        """
        return len(self.__alleles)
