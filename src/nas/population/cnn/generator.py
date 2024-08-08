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

import nas
from .individual import Individual


class Generator:
    """
    Generates a population of individuals based on a given seed.

    Attributes:
        __seed          (Seed): The seed for generating individuals.
        calculate_mem   (bool): Flag indicating whether to calculate mem.
        calculate_mua   (bool): Flag indicating whether to calculate mua.

    """

    def __init__(self, graph_controller, container):
        """
        Initializes new Generator object.

        Parameters:
            seed (nas.population.cnn.seed.Seed): The seed for generating individuals.
        """
        if not (isinstance(graph_controller, nas.population.cnn.graph_controller.GraphController)):
            raise ValueError("Not valid seed.")

        self.__graph_controller = graph_controller
        self.__container = container
        self.calculate_mem = True
        self.calculate_mua = True

    def get(self, num_of_individuals):
        """
        Generates a population of individuals.

        Parameters:
            num_of_individuals (int): The number of individuals to generate.

        Returns:
            list: List of generated individuals.

        Raises:
            ValueError: If num_of_individuals is not a positive integer.
        """
        if not (isinstance(num_of_individuals, int)) or num_of_individuals <= 0:
            raise ValueError(
                "Number of individuals to generate must be a positive integer."
            )

        population = []
        for _ in range(num_of_individuals):
            i = Individual(graph_controller=self.__graph_controller, container = self.__container)
            i.calculate_mem = self.calculate_mem
            i.calculate_mua = self.calculate_mua
            population.append(i)
        return population
