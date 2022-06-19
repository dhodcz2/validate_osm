import os

dir = os.getcwd()
os.chdir(os.path.dirname(__file__))
from .pfuncs import (
    get_strings,
    get_string,
    get_lengths,
    get_length,
    get_bound,
    get_bounds,
    get_claim,
)

os.chdir(dir)
