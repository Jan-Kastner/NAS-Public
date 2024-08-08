"""
File: searcher.py
Author: Jan Kastner
Date: February 29, 2024
"""

import os
import sys

# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))

# Append the current directory to sys.path
sys.path.append(current_directory)

from .template import Template
from .graph_controller import GraphController
from .generator import Generator
from .net import Net
