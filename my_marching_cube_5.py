"""Provides a function for performing 3D Marching Cubes"""

import numpy as np
import math
import time

from utils_3d import V3, Tri, Mesh, make_obj
from common import adapt
from larmor_radius import larmor_radius
from read_file import read_file
from threed_dda import threed_dda_2
from vertices_values import calculate_vertices_values as cvv

# My convention for vertices is:
VERTICES = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]

# My convention for the edges
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

# Table driven approach to the 256 combinations. Pro-tip, don't write this by hand, copy mine!
# See marching_cubes_gen.py for how I generated these.
# Each index is the bitwise representation of what is solid.
# Each value is a list of triples indicating what edges are used for that triangle
# (Recall each edge of the cell may become a vertex in the output boundary)
cases = [[],
 [[8, 0, 3]],
 [[1, 0, 9]],
 [[8, 1, 3], [8, 9, 1]],
 [[10, 2, 1]],
 [[8, 0, 3], [1, 10, 2]],
 [[9, 2, 0], [9, 10, 2]],
 [[3, 8, 2], [2, 8, 10], [10, 8, 9]],
 [[3, 2, 11]],
 [[0, 2, 8], [2, 11, 8]],
 [[1, 0, 9], [2, 11, 3]],
 [[2, 9, 1], [11, 9, 2], [8, 9, 11]],
 [[3, 10, 11], [3, 1, 10]],
 [[1, 10, 0], [0, 10, 8], [8, 10, 11]],
 [[0, 11, 3], [9, 11, 0], [10, 11, 9]],
 [[8, 9, 11], [11, 9, 10]],
 [[7, 4, 8]],
 [[3, 7, 0], [7, 4, 0]],
 [[7, 4, 8], [9, 1, 0]],
 [[9, 1, 4], [4, 1, 7], [7, 1, 3]],
 [[7, 4, 8], [2, 1, 10]],
 [[4, 3, 7], [4, 0, 3], [2, 1, 10]],
 [[2, 0, 10], [0, 9, 10], [7, 4, 8]],
 [[9, 10, 4], [4, 10, 3], [3, 10, 2], [4, 3, 7]],
 [[4, 8, 7], [3, 2, 11]],
 [[7, 4, 11], [11, 4, 2], [2, 4, 0]],
 [[1, 0, 9], [2, 11, 3], [8, 7, 4]],
 [[2, 11, 1], [1, 11, 9], [9, 11, 7], [9, 7, 4]],
 [[10, 11, 1], [11, 3, 1], [4, 8, 7]],
 [[4, 0, 7], [7, 0, 10], [0, 1, 10], [7, 10, 11]],
 [[7, 4, 8], [0, 11, 3], [9, 11, 0], [10, 11, 9]],
 [[4, 11, 7], [9, 11, 4], [10, 11, 9]],
 [[9, 4, 5]],
 [[9, 4, 5], [0, 3, 8]],
 [[0, 5, 1], [0, 4, 5]],
 [[4, 3, 8], [5, 3, 4], [1, 3, 5]],
 [[5, 9, 4], [10, 2, 1]],
 [[8, 0, 3], [1, 10, 2], [4, 5, 9]],
 [[10, 4, 5], [2, 4, 10], [0, 4, 2]],
 [[3, 10, 2], [8, 10, 3], [5, 10, 8], [4, 5, 8]],
 [[9, 4, 5], [11, 3, 2]],
 [[11, 0, 2], [11, 8, 0], [9, 4, 5]],
 [[5, 1, 4], [1, 0, 4], [11, 3, 2]],
 [[5, 1, 4], [4, 1, 11], [1, 2, 11], [4, 11, 8]],
 [[3, 10, 11], [3, 1, 10], [5, 9, 4]],
 [[9, 4, 5], [1, 10, 0], [0, 10, 8], [8, 10, 11]],
 [[5, 0, 4], [11, 0, 5], [11, 3, 0], [10, 11, 5]],
 [[5, 10, 4], [4, 10, 8], [8, 10, 11]],
 [[9, 7, 5], [9, 8, 7]],
 [[0, 5, 9], [3, 5, 0], [7, 5, 3]],
 [[8, 7, 0], [0, 7, 1], [1, 7, 5]],
 [[7, 5, 3], [3, 5, 1]],
 [[7, 5, 8], [5, 9, 8], [2, 1, 10]],
 [[10, 2, 1], [0, 5, 9], [3, 5, 0], [7, 5, 3]],
 [[8, 2, 0], [5, 2, 8], [10, 2, 5], [7, 5, 8]],
 [[2, 3, 10], [10, 3, 5], [5, 3, 7]],
 [[9, 7, 5], [9, 8, 7], [11, 3, 2]],
 [[0, 2, 9], [9, 2, 7], [7, 2, 11], [9, 7, 5]],
 [[3, 2, 11], [8, 7, 0], [0, 7, 1], [1, 7, 5]],
 [[11, 1, 2], [7, 1, 11], [5, 1, 7]],
 [[3, 1, 11], [11, 1, 10], [8, 7, 9], [9, 7, 5]],
 [[11, 7, 0], [7, 5, 0], [5, 9, 0], [10, 11, 0], [1, 10, 0]],
 [[0, 5, 10], [0, 7, 5], [0, 8, 7], [0, 10, 11], [0, 11, 3]],
 [[10, 11, 5], [11, 7, 5]],
 [[5, 6, 10]],
 [[8, 0, 3], [10, 5, 6]],
 [[0, 9, 1], [5, 6, 10]],
 [[8, 1, 3], [8, 9, 1], [10, 5, 6]],
 [[1, 6, 2], [1, 5, 6]],
 [[6, 2, 5], [2, 1, 5], [8, 0, 3]],
 [[5, 6, 9], [9, 6, 0], [0, 6, 2]],
 [[5, 8, 9], [2, 8, 5], [3, 8, 2], [6, 2, 5]],
 [[3, 2, 11], [10, 5, 6]],
 [[0, 2, 8], [2, 11, 8], [5, 6, 10]],
 [[3, 2, 11], [0, 9, 1], [10, 5, 6]],
 [[5, 6, 10], [2, 9, 1], [11, 9, 2], [8, 9, 11]],
 [[11, 3, 6], [6, 3, 5], [5, 3, 1]],
 [[11, 8, 6], [6, 8, 1], [1, 8, 0], [6, 1, 5]],
 [[5, 0, 9], [6, 0, 5], [3, 0, 6], [11, 3, 6]],
 [[6, 9, 5], [11, 9, 6], [8, 9, 11]],
 [[7, 4, 8], [6, 10, 5]],
 [[3, 7, 0], [7, 4, 0], [10, 5, 6]],
 [[7, 4, 8], [6, 10, 5], [9, 1, 0]],
 [[5, 6, 10], [9, 1, 4], [4, 1, 7], [7, 1, 3]],
 [[1, 6, 2], [1, 5, 6], [7, 4, 8]],
 [[6, 1, 5], [2, 1, 6], [0, 7, 4], [3, 7, 0]],
 [[4, 8, 7], [5, 6, 9], [9, 6, 0], [0, 6, 2]],
 [[2, 3, 9], [3, 7, 9], [7, 4, 9], [6, 2, 9], [5, 6, 9]],
 [[2, 11, 3], [7, 4, 8], [10, 5, 6]],
 [[6, 10, 5], [7, 4, 11], [11, 4, 2], [2, 4, 0]],
 [[1, 0, 9], [8, 7, 4], [3, 2, 11], [5, 6, 10]],
 [[1, 2, 9], [9, 2, 11], [9, 11, 4], [4, 11, 7], [5, 6, 10]],
 [[7, 4, 8], [11, 3, 6], [6, 3, 5], [5, 3, 1]],
 [[11, 0, 1], [11, 4, 0], [11, 7, 4], [11, 1, 5], [11, 5, 6]],
 [[6, 9, 5], [0, 9, 6], [11, 0, 6], [3, 0, 11], [4, 8, 7]],
 [[5, 6, 9], [9, 6, 11], [9, 11, 7], [9, 7, 4]],
 [[4, 10, 9], [4, 6, 10]],
 [[10, 4, 6], [10, 9, 4], [8, 0, 3]],
 [[1, 0, 10], [10, 0, 6], [6, 0, 4]],
 [[8, 1, 3], [6, 1, 8], [6, 10, 1], [4, 6, 8]],
 [[9, 2, 1], [4, 2, 9], [6, 2, 4]],
 [[3, 8, 0], [9, 2, 1], [4, 2, 9], [6, 2, 4]],
 [[0, 4, 2], [2, 4, 6]],
 [[8, 2, 3], [4, 2, 8], [6, 2, 4]],
 [[4, 10, 9], [4, 6, 10], [2, 11, 3]],
 [[11, 8, 2], [2, 8, 0], [6, 10, 4], [4, 10, 9]],
 [[2, 11, 3], [1, 0, 10], [10, 0, 6], [6, 0, 4]],
 [[8, 4, 1], [4, 6, 1], [6, 10, 1], [11, 8, 1], [2, 11, 1]],
 [[3, 1, 11], [11, 1, 4], [1, 9, 4], [11, 4, 6]],
 [[6, 11, 1], [11, 8, 1], [8, 0, 1], [4, 6, 1], [9, 4, 1]],
 [[3, 0, 11], [11, 0, 6], [6, 0, 4]],
 [[4, 11, 8], [4, 6, 11]],
 [[6, 8, 7], [10, 8, 6], [9, 8, 10]],
 [[3, 7, 0], [0, 7, 10], [7, 6, 10], [0, 10, 9]],
 [[1, 6, 10], [0, 6, 1], [7, 6, 0], [8, 7, 0]],
 [[10, 1, 6], [6, 1, 7], [7, 1, 3]],
 [[9, 8, 1], [1, 8, 6], [6, 8, 7], [1, 6, 2]],
 [[9, 7, 6], [9, 3, 7], [9, 0, 3], [9, 6, 2], [9, 2, 1]],
 [[7, 6, 8], [8, 6, 0], [0, 6, 2]],
 [[3, 6, 2], [3, 7, 6]],
 [[3, 2, 11], [6, 8, 7], [10, 8, 6], [9, 8, 10]],
 [[7, 9, 0], [7, 10, 9], [7, 6, 10], [7, 0, 2], [7, 2, 11]],
 [[0, 10, 1], [6, 10, 0], [8, 6, 0], [7, 6, 8], [2, 11, 3]],
 [[1, 6, 10], [7, 6, 1], [11, 7, 1], [2, 11, 1]],
 [[1, 9, 6], [9, 8, 6], [8, 7, 6], [3, 1, 6], [11, 3, 6]],
 [[9, 0, 1], [11, 7, 6]],
 [[0, 11, 3], [6, 11, 0], [7, 6, 0], [8, 7, 0]],
 [[7, 6, 11]],
 [[11, 6, 7]],
 [[3, 8, 0], [11, 6, 7]],
 [[1, 0, 9], [6, 7, 11]],
 [[1, 3, 9], [3, 8, 9], [6, 7, 11]],
 [[10, 2, 1], [6, 7, 11]],
 [[10, 2, 1], [3, 8, 0], [6, 7, 11]],
 [[9, 2, 0], [9, 10, 2], [11, 6, 7]],
 [[11, 6, 7], [3, 8, 2], [2, 8, 10], [10, 8, 9]],
 [[2, 6, 3], [6, 7, 3]],
 [[8, 6, 7], [0, 6, 8], [2, 6, 0]],
 [[7, 2, 6], [7, 3, 2], [1, 0, 9]],
 [[8, 9, 7], [7, 9, 2], [2, 9, 1], [7, 2, 6]],
 [[6, 1, 10], [7, 1, 6], [3, 1, 7]],
 [[8, 0, 7], [7, 0, 6], [6, 0, 1], [6, 1, 10]],
 [[7, 3, 6], [6, 3, 9], [3, 0, 9], [6, 9, 10]],
 [[7, 8, 6], [6, 8, 10], [10, 8, 9]],
 [[8, 11, 4], [11, 6, 4]],
 [[11, 0, 3], [6, 0, 11], [4, 0, 6]],
 [[6, 4, 11], [4, 8, 11], [1, 0, 9]],
 [[1, 3, 9], [9, 3, 6], [3, 11, 6], [9, 6, 4]],
 [[8, 11, 4], [11, 6, 4], [1, 10, 2]],
 [[1, 10, 2], [11, 0, 3], [6, 0, 11], [4, 0, 6]],
 [[2, 9, 10], [0, 9, 2], [4, 11, 6], [8, 11, 4]],
 [[3, 4, 9], [3, 6, 4], [3, 11, 6], [3, 9, 10], [3, 10, 2]],
 [[3, 2, 8], [8, 2, 4], [4, 2, 6]],
 [[2, 4, 0], [6, 4, 2]],
 [[0, 9, 1], [3, 2, 8], [8, 2, 4], [4, 2, 6]],
 [[1, 2, 9], [9, 2, 4], [4, 2, 6]],
 [[10, 3, 1], [4, 3, 10], [4, 8, 3], [6, 4, 10]],
 [[10, 0, 1], [6, 0, 10], [4, 0, 6]],
 [[3, 10, 6], [3, 9, 10], [3, 0, 9], [3, 6, 4], [3, 4, 8]],
 [[9, 10, 4], [10, 6, 4]],
 [[9, 4, 5], [7, 11, 6]],
 [[9, 4, 5], [7, 11, 6], [0, 3, 8]],
 [[0, 5, 1], [0, 4, 5], [6, 7, 11]],
 [[11, 6, 7], [4, 3, 8], [5, 3, 4], [1, 3, 5]],
 [[1, 10, 2], [9, 4, 5], [6, 7, 11]],
 [[8, 0, 3], [4, 5, 9], [10, 2, 1], [11, 6, 7]],
 [[7, 11, 6], [10, 4, 5], [2, 4, 10], [0, 4, 2]],
 [[8, 2, 3], [10, 2, 8], [4, 10, 8], [5, 10, 4], [11, 6, 7]],
 [[2, 6, 3], [6, 7, 3], [9, 4, 5]],
 [[5, 9, 4], [8, 6, 7], [0, 6, 8], [2, 6, 0]],
 [[7, 3, 6], [6, 3, 2], [4, 5, 0], [0, 5, 1]],
 [[8, 1, 2], [8, 5, 1], [8, 4, 5], [8, 2, 6], [8, 6, 7]],
 [[9, 4, 5], [6, 1, 10], [7, 1, 6], [3, 1, 7]],
 [[7, 8, 6], [6, 8, 0], [6, 0, 10], [10, 0, 1], [5, 9, 4]],
 [[3, 0, 10], [0, 4, 10], [4, 5, 10], [7, 3, 10], [6, 7, 10]],
 [[8, 6, 7], [10, 6, 8], [5, 10, 8], [4, 5, 8]],
 [[5, 9, 6], [6, 9, 11], [11, 9, 8]],
 [[11, 6, 3], [3, 6, 0], [0, 6, 5], [0, 5, 9]],
 [[8, 11, 0], [0, 11, 5], [5, 11, 6], [0, 5, 1]],
 [[6, 3, 11], [5, 3, 6], [1, 3, 5]],
 [[10, 2, 1], [5, 9, 6], [6, 9, 11], [11, 9, 8]],
 [[3, 11, 0], [0, 11, 6], [0, 6, 9], [9, 6, 5], [1, 10, 2]],
 [[0, 8, 5], [8, 11, 5], [11, 6, 5], [2, 0, 5], [10, 2, 5]],
 [[11, 6, 3], [3, 6, 5], [3, 5, 10], [3, 10, 2]],
 [[3, 9, 8], [6, 9, 3], [5, 9, 6], [2, 6, 3]],
 [[9, 6, 5], [0, 6, 9], [2, 6, 0]],
 [[6, 5, 8], [5, 1, 8], [1, 0, 8], [2, 6, 8], [3, 2, 8]],
 [[2, 6, 1], [6, 5, 1]],
 [[6, 8, 3], [6, 9, 8], [6, 5, 9], [6, 3, 1], [6, 1, 10]],
 [[1, 10, 0], [0, 10, 6], [0, 6, 5], [0, 5, 9]],
 [[3, 0, 8], [6, 5, 10]],
 [[10, 6, 5]],
 [[5, 11, 10], [5, 7, 11]],
 [[5, 11, 10], [5, 7, 11], [3, 8, 0]],
 [[11, 10, 7], [10, 5, 7], [0, 9, 1]],
 [[5, 7, 10], [10, 7, 11], [9, 1, 8], [8, 1, 3]],
 [[2, 1, 11], [11, 1, 7], [7, 1, 5]],
 [[3, 8, 0], [2, 1, 11], [11, 1, 7], [7, 1, 5]],
 [[2, 0, 11], [11, 0, 5], [5, 0, 9], [11, 5, 7]],
 [[2, 9, 5], [2, 8, 9], [2, 3, 8], [2, 5, 7], [2, 7, 11]],
 [[10, 3, 2], [5, 3, 10], [7, 3, 5]],
 [[10, 0, 2], [7, 0, 10], [8, 0, 7], [5, 7, 10]],
 [[0, 9, 1], [10, 3, 2], [5, 3, 10], [7, 3, 5]],
 [[7, 8, 2], [8, 9, 2], [9, 1, 2], [5, 7, 2], [10, 5, 2]],
 [[3, 1, 7], [7, 1, 5]],
 [[0, 7, 8], [1, 7, 0], [5, 7, 1]],
 [[9, 5, 0], [0, 5, 3], [3, 5, 7]],
 [[5, 7, 9], [7, 8, 9]],
 [[4, 10, 5], [8, 10, 4], [11, 10, 8]],
 [[3, 4, 0], [10, 4, 3], [10, 5, 4], [11, 10, 3]],
 [[1, 0, 9], [4, 10, 5], [8, 10, 4], [11, 10, 8]],
 [[4, 3, 11], [4, 1, 3], [4, 9, 1], [4, 11, 10], [4, 10, 5]],
 [[1, 5, 2], [2, 5, 8], [5, 4, 8], [2, 8, 11]],
 [[5, 4, 11], [4, 0, 11], [0, 3, 11], [1, 5, 11], [2, 1, 11]],
 [[5, 11, 2], [5, 8, 11], [5, 4, 8], [5, 2, 0], [5, 0, 9]],
 [[5, 4, 9], [2, 3, 11]],
 [[3, 4, 8], [2, 4, 3], [5, 4, 2], [10, 5, 2]],
 [[5, 4, 10], [10, 4, 2], [2, 4, 0]],
 [[2, 8, 3], [4, 8, 2], [10, 4, 2], [5, 4, 10], [0, 9, 1]],
 [[4, 10, 5], [2, 10, 4], [1, 2, 4], [9, 1, 4]],
 [[8, 3, 4], [4, 3, 5], [5, 3, 1]],
 [[1, 5, 0], [5, 4, 0]],
 [[5, 0, 9], [3, 0, 5], [8, 3, 5], [4, 8, 5]],
 [[5, 4, 9]],
 [[7, 11, 4], [4, 11, 9], [9, 11, 10]],
 [[8, 0, 3], [7, 11, 4], [4, 11, 9], [9, 11, 10]],
 [[0, 4, 1], [1, 4, 11], [4, 7, 11], [1, 11, 10]],
 [[10, 1, 4], [1, 3, 4], [3, 8, 4], [11, 10, 4], [7, 11, 4]],
 [[9, 4, 1], [1, 4, 2], [2, 4, 7], [2, 7, 11]],
 [[1, 9, 2], [2, 9, 4], [2, 4, 11], [11, 4, 7], [3, 8, 0]],
 [[11, 4, 7], [2, 4, 11], [0, 4, 2]],
 [[7, 11, 4], [4, 11, 2], [4, 2, 3], [4, 3, 8]],
 [[10, 9, 2], [2, 9, 7], [7, 9, 4], [2, 7, 3]],
 [[2, 10, 7], [10, 9, 7], [9, 4, 7], [0, 2, 7], [8, 0, 7]],
 [[10, 4, 7], [10, 0, 4], [10, 1, 0], [10, 7, 3], [10, 3, 2]],
 [[8, 4, 7], [10, 1, 2]],
 [[4, 1, 9], [7, 1, 4], [3, 1, 7]],
 [[8, 0, 7], [7, 0, 1], [7, 1, 9], [7, 9, 4]],
 [[0, 7, 3], [0, 4, 7]],
 [[8, 4, 7]],
 [[9, 8, 10], [10, 8, 11]],
 [[3, 11, 0], [0, 11, 9], [9, 11, 10]],
 [[0, 10, 1], [8, 10, 0], [11, 10, 8]],
 [[11, 10, 3], [10, 1, 3]],
 [[1, 9, 2], [2, 9, 11], [11, 9, 8]],
 [[9, 2, 1], [11, 2, 9], [3, 11, 9], [0, 3, 9]],
 [[8, 2, 0], [8, 11, 2]],
 [[11, 2, 3]],
 [[2, 8, 3], [10, 8, 2], [9, 8, 10]],
 [[0, 2, 9], [2, 10, 9]],
 [[3, 2, 8], [8, 2, 10], [8, 10, 1], [8, 1, 0]],
 [[1, 2, 10]],
 [[3, 1, 8], [1, 9, 8]],
 [[9, 0, 1]],
 [[3, 0, 8]],
 []]


# Find how many points there in each cube.
# def find_location_of_cube(x, y, z, d = 1):
#     x_2 = (xmax + x) // d + 1
#     y_2 = ((ymax + y) // d) * length_x
#     z_2 = ((zmax + z) // d) * length_x * length_y
#     index = x_2 + y_2 + z_2
#     cubes[index] = cubes[index] + 1

# Marching cubes method.
def marching_cubes_3d_single_cell(cube, x, y, z):
    # Determine which case we are.
    case = sum(2**v for v in range(8) if cube[v] > 0)
    # Ok, what faces do we need (in terms of edges)
    faces = cases[case]

    def edge_to_boundary_vertex(edge):
        """Returns the vertex in the middle of the specified edge"""
        # Find the two vertices specified by this edge, and interpolate between
        # them according to adapt, as in the 2d case
        v0, v1 = EDGES[edge]
        f0 = cube[v0]
        f1 = cube[v1]
        t0 = 1 - adapt(f0, f1)
        t1 = 1 - t0
        vert_pos0 = VERTICES[v0]
        vert_pos1 = VERTICES[v1]
        return V3(x + vert_pos0[0] * t0 + vert_pos1[0] * t1,
                  y + vert_pos0[1] * t0 + vert_pos1[1] * t1,
                  z + vert_pos0[2] * t0 + vert_pos1[2] * t1)

    output_verts = []
    output_tris = []

    for face in faces:
        # For each face, find the vertices of that face, and output it.
        # We make no effort to re-use vertices between multiple faces,
        # A fancier implementation might do so.
        edges = face
        verts = list(map(edge_to_boundary_vertex, edges))
        next_vert_index = len(output_verts) + 1
        tri = Tri(
            next_vert_index,
            next_vert_index+1,
            next_vert_index+2,
        )
        output_verts.extend(verts)
        output_tris.append(tri)
    return Mesh(output_verts, output_tris)

# Calculate the vertex value.
def make_3d_cubes(maximum):
    mesh = Mesh()
    for z in range(0, int(length_z)):
        for y in range(0, int(length_y)):
            for x in range(0, int(length_x)):
                n = x + length_x * y + length_x * length_y * z
                middle_cubles = [n - length_x - 1,
                 n - length_x,
                 n - length_x + 1,
                 n - 1,
                 n,
                 n + 1,
                 n + length_x - 1,
                 n + length_x,
                 n + length_x + 1]
                down_layer_cubes = [middle_cubles[i] - length_x * length_y for i in range(len(middle_cubles))]
                up_layer_cubes = [middle_cubles[i] + length_x * length_y for i in range(len(middle_cubles))]
                surround_cubes = down_layer_cubes + middle_cubles + up_layer_cubes
                s = surround_cubes

                values = []
                for i in range(len(s)):
                    if s[i] in vertices.keys():
                        mapping = [1 - j / maximum for j in vertices[s[i]]]
                        values.append(mapping)
                    else:
                        values.append([0] * 8)

                cube_list = [[0, 1, 3, 4, 9, 10, 12, 13],
                             [1, 2, 4, 5, 10, 11, 13, 14],
                             [4, 5, 7, 8, 13, 14, 16, 17],
                             [3, 4, 6, 7, 12, 13, 15, 16],
                             [9, 10, 12, 13, 18, 19, 21, 22],
                             [10, 11, 13, 14, 19, 20, 22, 23],
                             [13, 14, 16, 17, 22, 23, 25, 26],
                             [12, 13, 15, 16, 21, 22, 24, 25]]
                vertex_list = [6, 7, 5, 4, 2, 3, 1, 0]
                vertex_2 = []
                for a in cube_list:
                    # print([values[b][c] for b, c in zip(a, vertex_list)])
                    vertex = max([values[b][c] for b, c in zip(a, vertex_list)])
                    # vertex_value_list = [values[b][c] for b, c in zip(a, vertex_list)]
                    # vertex = sum(vertex_value_list)/8
                    vertex_2.append(vertex)
                cell_mesh = marching_cubes_3d_single_cell(vertex_2, x - xmax, y - ymax, z - zmax)
                mesh.extend(cell_mesh)
    return mesh


# For each cube, evaluate independently. Second method.
# def make_3d_cubes():
#     mesh = Mesh()
#     for z in range(0, int(length_z)):
#         for y in range(0, int(length_y)):
#             for x in range(1, int(length_x) + 1):
#                 n = x + length_x * y + length_x * length_y * z
#                 middle_cubles = [n - length_x - 1,
#                  n - length_x,
#                  n - length_x + 1,
#                  n - 1,
#                  n,
#                  n + 1,
#                  n + length_x - 1,
#                  n + length_x,
#                  n + length_x + 1]
#                 down_layer_cubes = [middle_cubles[i] - length_x * length_y for i in range(len(middle_cubles))]
#                 up_layer_cubes = [middle_cubles[i] + length_x * length_y for i in range(len(middle_cubles))]
#                 surround_cubes = down_layer_cubes + middle_cubles + up_layer_cubes

#                 values = [None] * len(surround_cubes)
#                 vertex = [0] * 8
#                 for i in range(len(surround_cubes)):
#                     if cubes.get(surround_cubes[i]):
#                         values[i] = cubes[surround_cubes[i]]
#                     else:
#                         values[i] = 0

#                 cube_list = [[0, 1, 3, 4, 9, 10, 12, 13],
#                  [1, 2, 4, 5, 10, 11, 13, 14],
#                  [4, 5, 7, 8, 13, 14, 16, 17],
#                  [3, 4, 6, 7, 12, 13, 15, 16],
#                  [9, 10, 12, 13, 18, 19, 21, 22],
#                  [10, 11, 13, 14, 19, 20, 22, 23],
#                  [13, 14, 16, 17, 22, 23, 25, 26],
#                  [12, 13, 15, 16, 21, 22, 24, 25]]

#                 for i in range(8):
#                     for a in cube_list[i]:
#                         vertex[i] += values[a]
#                     vertex[i] = vertex[i] * 0.125
#                 cell_mesh = marching_cubes_3d_single_cell(vertex, x - 1, y, z)
#                 mesh.extend(cell_mesh)
#     return mesh

def make_a_obj(filename, maximum):
    """Writes an obj file containing a sphere meshed via marching cubes"""
    mesh = make_3d_cubes(maximum)
    with open(filename, "w") as f:
        make_obj(f, mesh)

if __name__ == "__main__":
    # Open the file.
    file_name = 'data/FFHR-d1A_R3.50_20181105_extract.dat'

    xmin, xmax = -192, 192
    ymin, ymax = -192, 192
    zmin, zmax = -56, 56

    length_x = xmax - xmin
    length_y = ymax - ymin
    length_z = zmax - zmin

    # Initialize each cube with 0.
    # cubes = {}
    # for z in range(0, int(length_z)):
    #     for y in range(0, int(length_y)):
    #         for x in range(1, int(length_x + 1)):
    #             index = x + y * length_x + z * length_x * length_y
    #             cubes[index] = 0

    # Find the cube which has point in it and set the value of it as 1.
    start1 = time.time()
    lines = read_file(file_name)
    time1 = time.time() - start1
    print(f"time of running 'read_file' is {time1:.2f} s.")

    scale = 8
    cubes = {}
    vertices = {}
    # all_points = []

    time2, time3, time4 = 0, 0, 0

    for line in lines:
        all_points_of_a_line = []
        start2 = time.time()
        modified_points = larmor_radius(line)
        time2 += time.time() - start2

        start3 = time.time()
        all_points_of_a_line.append(threed_dda_2(modified_points, scale))
        time3 += time.time() - start3

        start4 = time.time()
        for tube_segment in all_points_of_a_line:
            for line_segment in tube_segment:
                cvv(xmax, ymax, zmax, line_segment, cubes, vertices)
        time4 += time.time() - start4

    over_distance = []
    for u, v in vertices.items():
        for p in v:
            if p > 1.733:
                over_distance.append(p)
                print(f'index {u}: distance is over, {p}')
    maximum = max(over_distance)

    print(f"time of running 'larmor_radius' is {time2:.2f} s.")
    print(f"time of running '3d_dda' is {time3:.2f} s.")
    print(f"time of running 'vertices_values' is {time4:.2f} s.")

    # for point in all_points:
    #     try:
    #         find_location_of_cube(point[0], point[1], point[2])
    #     except TypeError:
    #         print(point)

    start5 = time.time()
    make_a_obj(f"output_5_scale_{scale}.obj", maximum)
    time5 = time.time() - start5
    print(f"time of running 'marching_cubes' is {time5:.2f} s.")