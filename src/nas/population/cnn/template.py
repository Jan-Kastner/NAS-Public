"""
File: generator.py
Author: Jan Kastner
Date: February 29, 2024

Description:    
    Template:  Represents template for genotype of the individual.
"""


class Template:
    """
    Represents a template for CGP graph.

    Attributes:
        self.grid (list): Stores layer classes.
    """

    def __init__(self, rows, cols, L_back):
        """
        Sets the template configuration.

        Parameters:
            rows (list): List of layer classes.

            cols (list): List of column width for each row.

            L_back (int): The number of rows preceding the j-th row from which an input can be selected
            for a node in the j-th row.

        Raises:
            ValueError: If any of the following conditions are met:
                - First layer can't be a pooling layer.
                - Not a valid configuration. Maximum L_back pooling layers in a row is supported.
                - Invalid layer class.
                - Number of columns in row isn't specified by integer.
                - Template doesn't contain at least 4 rows.
        """
        self.grid = []

        self.__rows = rows

        if not self.check_fully_connected_layers_sequence(rows):
            raise ValueError(
                "Invalid sequence: Fully connected layers ('FC') can only occur at the end of the model."
            )

        if not isinstance(L_back, int) or L_back <= 0:
            raise ValueError("L_back must be a positive integer.")

        self._L_back = min(L_back, len(rows))

        if rows[0] == "PL":
            raise ValueError("First layer can't be pooling layer.")

        if any(
            all(element == "PL" for element in rows[idx : self._L_back + 1 + idx])
            for idx in range(len(rows))
        ):
            raise ValueError(
                "Not valid configuration. Maximum L_back ('{}') pooling layers in row is supported.".format(
                    self._L_back
                )
            )

        if not all(x in ["CL", "PL", "FC"] for x in rows):
            raise ValueError(
                f"Invalid layer class specified in: {rows}. "
                f"Please choose from: {', '.join(['CL', 'PL', 'FC'])}."
            )

        if not all(isinstance(x, int) and 1 <= x for x in cols):
            raise ValueError(
                "Only positive integers are allowed to specify number of cols in row: '{}'.".format(
                    cols
                )
            )

        if not (len(rows) == len(cols) and len(rows) > 3):
            raise ValueError("Template must contain at least 4 rows.")

        self.grid = []

        # Add input layer to the grid
        self.grid.append(["INPUT"])

        # Add each layer class with its respective number of columns to the grid
        for layer_class, col in zip(rows, cols):
            self.grid.append([layer_class] * col)

        # Add output layer to the grid
        self.grid.append(["OUTPUT"])

        self._cols = [1] + cols + [1]

    def check_fully_connected_layers_sequence(self, rows):
        fc_occurred = False
        for i, row in enumerate(rows):
            if row == "FC":
                if fc_occurred and i > 0 and rows[i - 1] != "FC":
                    return False
                fc_occurred = True
            elif fc_occurred:
                return False
        return True

    @property
    def cols(self):
        return self._cols

    @property
    def L_back(self):
        return self._L_back
    
    @property
    def specified_conv_class(self):
        return 'CL' in self.__rows
    
    @property
    def specified_pooling_class(self):
        return 'PL' in self.__rows
    
    @property
    def specified_fully_connected_class(self):
        return 'FC' in self.__rows
