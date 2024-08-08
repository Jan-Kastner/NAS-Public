"""
File: generator.py
Author: Jan Kastner
Date: February 29, 2024

Description:    
    Seed:  Generates and muatates nodes for CGP.
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

from encoding.CGP_graph import Node as CGPN
import random
import nas


class GraphController:
    LAYER_TYPE_IDX = 0
    INPUT1_IDX = 1
    INPUT2_IDX = 2
    """
    Generates and muatates nodes for CGP.

    Attributes:
        _template (list): List of layer classes what represents template for CGPgraph.

        _cols (list): List of column width for each row.

        __L_back (int): The number of rows preceding the j-th row from which an input can be selected 
        for a node in the j-th row.

        __ML_probability (int, float): The probability with which the pooling layer can be replaced by 
        the merging layer.

        __num_of_nodes (int): The number of CGPnodes in CGP graph.

        __abs_to_r_c (list): List that maps the absolute position to its corresponding row and column 
        positions in the grid of CGP graph.

        __r_c_to_abs (list): List that maps the row and column position of CGPnode in the grid of CGP 
        graph to its corresponding absolute position.

        __seed (dict): Dictionary storing types corresponding to layer class. It has following structure:
                            self.__types = {
                                'INPUT': [],
                                'CL': [],
                                'PL': [],
                                'FC': [],
                                'ML': [],
                                'OUTPUT': []}
    """

    def __init__(self, template, ML_probability, container):
        """
        Sets the seed configuration.

        Parameters:
            L_back (int): The number of rows preceding the j-th row from which an input can be
            selected for a node in the j-th row.

            ML_probability (int, float): The probability with which the pooling layer can be replaced by the
            merging layer.

            rows (list): List of layer classes. One class is assigned to one row.

            cols (list): List of column width for each row.

        Raises:
            ValueError: If any of the following conditions are met:
                - L_back is not positive integer.
                - ML_probability is not integer or float in interval <0,1>
                - The provided 'container' is not valid.
                - The provided 'template' is not valid.
                - No convolutional layers were specified.
                - No pooling layers were specified.
                - No fully connected layers were specified.
                - No merging layers were specified.
                - No output layer was specified.
        """

        if not (isinstance(container, nas.population.Container)):
            raise ValueError("The provided 'container' is not valid.")

        if not (isinstance(template, nas.population.cnn.template.Template)):
            raise ValueError("The provided 'template' is not valid.")

        if not (isinstance(ML_probability, (int, float)) and 0 <= ML_probability <= 1):
            raise ValueError("ML_probability must be in the interval <0, 1>.")
        
        if template.specified_conv_class and not container.specified_conv_layers:
            raise ValueError("No convolutional layers were specified.")
        
        if template.specified_pooling_class and not container.specified_pooling_layers:
            raise ValueError("No pooling layers were specified.")

        if template.specified_fully_connected_class and not container.specified_fully_connected_layers:
            raise ValueError("No fully connected layers were specified.")

        if ML_probability > 0 and not container.specified_merging_layers:
            raise ValueError("No merging layers were specified.")
        
        if not container.specified_output_layer:
            raise ValueError("No output layer was specified.")

        self._template = template
        self.__container = container
        self.__L_back = self._template.L_back
        self.__ML_probability = ML_probability
        self.__set_transfers()

    def __set_transfers(self):
        """
        Sets lists to effectively get row and col from absolute position of node in CGP graph and vice
        versa.
        """
        node_num = 0
        self.__abs_to_r_c = []
        self.__r_c_to_abs = []

        for row_idx, row in enumerate(self._template.grid):
            self.__r_c_to_abs.append([])
            for col_idx in range(len(row)):
                self.__abs_to_r_c.append([row_idx, col_idx])
                self.__r_c_to_abs[row_idx].append(node_num)
                node_num += 1

        self.__num_of_nodes = node_num

    def get_layer_type(self, n_position):
        """
        Gets the type of layer at the specified position in the CGP graph.
        With probability '__ML_probability' is "PL" (pooling layer) replaced with "ML" (merging layer)

        Parameters:
            n_position (int): The absolute position of the layer in the CGP graph.

        Returns:
            int: The type of layer at the specified position, None otherwise.
        """
        # If n_position is out of bounds, return None
        if n_position >= self.__num_of_nodes:
            return None

        n_row, n_col = self.__abs_to_r_c[n_position]

        n_layer_class = self._template.grid[n_row][n_col]

        # with probability 'ML_probability' replace pooling layer with merging layer
        if n_layer_class == "PL" or n_layer_class == "ML":
            if random.random() <= self.__ML_probability:
                n_layer_class = "ML"
            self._template.grid[n_row][n_col] = n_layer_class

        return random.choice(self.__container.types[n_layer_class])

    def get_layer_connection(self, n_position, input1=False, input2=False):
        """
        Gets the connection for the layer at the specified position in the CGP graph.

        Parameters:
            n_position (int): The absolute position of the layer in the CGP graph.

            input1, input2 (bool): Whether to get the first or second connection (true = get connection).

        Returns:
            int or None: The absolute position of the connected layer.
        """
        # input not specified
        if not (
            isinstance(input1, bool) and isinstance(input2, bool) and (input1 or input2)
        ):
            return None

        # node position out of range
        if n_position >= self.__num_of_nodes:
            return None

        n_row, n_col = self.__abs_to_r_c[n_position]

        n_layer_class = self._template.grid[n_row][n_col]

        # each layer except merging layer doesn't have second input
        if input2 and n_layer_class != "ML":
            return None

        # input layer doesn't have inputs
        if n_layer_class == "INPUT":
            return None

        first_row = n_row - self.__L_back
        last_row = n_row - 1

        if first_row < 0:
            first_row = 0

        while True:
            # position of node to be connected
            row = random.randint(first_row, last_row)
            col = random.randint(0, len(self._template.grid[row]) - 1)

            i_layer_class = self._template.grid[row][col]

            if i_layer_class == "INPUT":
                # input can't be directly connected to pooling layer or merging layer
                if n_layer_class == "PL" or n_layer_class == "ML":
                    continue

                return self.__r_c_to_abs[row][col]
            # pooling layers can't be directly connected
            if n_layer_class == "PL" and i_layer_class == "PL":
                continue

            return self.__r_c_to_abs[row][col]

    def nodes(self):
        """
        Generates CGPNodes based on the configured template.

        Returns:
            list: List of generated CGPNodes.
        """
        nodes = []

        # avoid input and output
        num_nodes = sum(self._template.cols[1:])
        # skip input
        for n_position in range(1, num_nodes):

            cgpn = CGPN(number_of_inputs=2)
            cgpn.set_type(self.get_layer_type(n_position))
            cgpn.set_input(
                self.INPUT1_IDX,
                self.get_layer_connection(n_position=n_position, input1=True),
            )
            cgpn.set_input(
                self.INPUT2_IDX,
                self.get_layer_connection(n_position=n_position, input2=True),
            )

            nodes.append(cgpn)

        return nodes

    def mutate_node(self, n_position, a_position, node):
        """
        Mutates the specified CGPNode at the given position and allele.

        Parameters:
            n_position (int): The absolute position of the CGPNode in the CGP graph.
            a_position (int): The position of the allele to mutate in the CGPNode.
            node (CGPNode): The CGPNode object to mutate.

        Returns:
            CGPNode or None: The mutated CGPNode if successful, None otherwise.
        """
        if not (isinstance(node, CGPN)):
            return None  # If node is not a CGPNode object, return None

        # Mutate the first input
        if a_position == self.INPUT1_IDX:
            node.set_input(
                a_position,
                self.get_layer_connection(n_position=n_position, input1=True),
            )
        # Mutate the second input
        elif a_position == self.INPUT2_IDX:
            node.set_input(
                a_position,
                self.get_layer_connection(n_position=n_position, input2=True),
            )
        else:
            # Mutate the type
            node.set_type(self.get_layer_type(n_position))
            # Mutate the second input
            node.set_input(
                self.INPUT2_IDX,
                self.get_layer_connection(n_position=n_position, input2=True),
            )

        return node

    def mutate_output(self):
        """
        Mutates the output connection.
        Returns:
            int: The mutated output connection value.
        """
        return self.get_layer_connection(
            n_position=sum(self._template.cols[1:]), input1=True
        )

    def inputs(self):
        """
        Returns the input connections.

        Returns:
            list: List containing the input connections.
        """
        return [0]

    def outputs(self):
        """
        Returns the output connections.

        Returns:
            list: List containing the output connections.
        """
        return [
            self.get_layer_connection(
                n_position=sum(self._template.cols[1:]), input1=True
            )
        ]

    @property
    def shape(self):
        """
        Property to access the shape.

        Returns:
            list: List containing the shape.
        """
        return self._template.cols[1:-1]