import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord

def map_spec(a):
    return a + 10

def map_block2D_test(cuda):
    def call(out, a, size):
        # FILL ME IN (roughly 4 lines)
    return call


SIZE = 5
out = np.zeros((SIZE, SIZE))
a = np.ones((SIZE, SIZE))

problem = CudaProblem(
    "Blocks 2D",
    map_block2D_test,
    [a],
    out,
    [SIZE],
    threadsperblock=Coord(3, 3),
    blockspergrid=Coord(2, 2),
    spec=map_spec,
)

problem.check()