# src/utils.py
import os
import sys

# This file centralizes the path manipulation to access parent directory modules
# We only need to modify sys.path once in this utility file

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now we can import from parent directory
from logger_setup import setup_logger

def get_logger(name):
    """
    Get a properly configured logger for the specified module name.
    This ensures all modules use the same logging configuration.
    """
    return setup_logger(name)