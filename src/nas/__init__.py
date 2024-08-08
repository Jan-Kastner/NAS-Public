"""
File: searcher.py
Author: Jan Kastner
Date: February 29, 2024

Description:
    NSGAE:  wraps the individual and provides method important for NSGA-II algorithm.
    Searcher: searches for neural network architecture. Searching is based on NSGA-II algorithm.
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

import opt
import population
from terminator import Terminator
