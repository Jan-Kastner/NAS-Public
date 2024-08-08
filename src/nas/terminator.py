"""
File: terminator.py
Author: Jan Kastner
Date: February 29, 2024

Description:    
    Termiantor: contains termination strategies used within NAS.
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

import time


class Terminator:
    """
    Represents terminators for Evolutionary Algorithm. Determine when to stop the evolutionary
    process based on either the maximum number of generations or the maximum runtime.

    Attributes:
        __end_time              (float): The end time for the EA run.
        __remaining_generations (int): The remaining number of generations.
    """

    def __init__(self, run_time=None, max_generations=None):
        """
        Sets the termination conditions for the EA run.

        Parameters:
            run_time (float, optioanl): The maximum runtime allowed for the EA run.

            max_generations (int, optional): The maximum number of generations allowed for the EA run.
        """
        self.__end_time = (
            run_time + time.time() if self.__positive_integer(run_time) else None
        )
        self.__remaining_generations = (
            max_generations if self.__positive_integer(max_generations) else None
        )

    def __positive_integer(self, value):
        """
        Checks if the given value is a positive integer.

        Parameters:
            value: The value to be checked.

        Returns:
            bool: True if the value is a positive integer, False otherwise.
        """
        return isinstance(value, int) and value > 0

    def generation_termination(self):
        """
        Determines whether the termination condition based on maximum generations is met.

        Returns:
            bool: True if the maximum number of generations is reached, False otherwise.

        Raises:
            ValueError: If the maximum number of generations is not specified.
        """
        if self.__remaining_generations is None:
            raise ValueError("Maximum number of generations not specified.")
        else:
            self.__remaining_generations -= 1
            return self.__remaining_generations < 0

    def time_termination(self):
        """
        Determines whether the termination condition based on runtime is met.

        Returns:
            bool: True if the runtime limit is reached, False otherwise.

        Raises:
            ValueError: If the runtime limit is not specified.
        """
        if self.__end_time is None:
            raise ValueError("Run time not specified.")
        else:
            return time.time() >= self.__end_time

    def time_generation_termination(self):
        """
        Determines whether the termination condition based on both maximum generations and runtime is met.

        Returns:
            bool: True if either the maximum generations or runtime limit is reached, False otherwise.

        Raises:
            ValueError: If the maximum number of generations or runtime limit is not specified.
        """
        if self.__remaining_generations is None:
            raise ValueError("Maximum number of generations not specified.")
        elif self.__end_time is None:
            raise ValueError("Run time not specified.")
        else:
            self.__remaining_generations -= 1
            return time.time() >= self.__end_time or self.__remaining_generations < 0

    @property
    def terminations(self):
        """
        Returns a list of termination conditions.

        Returns:
            list: A list of termination condition functions.
        """
        return [
            Terminator.generation_termination,
            Terminator.time_termination,
            Terminator.time_generation_termination,
        ]
