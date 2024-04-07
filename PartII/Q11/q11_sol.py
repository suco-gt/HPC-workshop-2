import numba
import numpy as np
import warnings
from lib import CudaProblem, Coord

warnings.filterwarnings(
    action="ignore", category=numba.NumbaPerformanceWarning, module="numba"
)

def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < a_size:
            shared_a[local_i] = a[i]
        if i < b_size:
            shared_b[local_i] = b[i]
        else:
            local_i2 = local_i - b_size
            i2 = i - b_size
            if i2 + TPB < a_size and local_i2 < b_size
                shared_a[TPB + local_i2] = a[i2 + TPB]
        cuda.syncthreads()
        
        acc = 0
        for k in range(b_size):
            if i + k < a_size:
                acc += shared_a[local_i + k] * shared_b[k]

        if i < a_size:
            out[i] = acc

    return call


# Test 1

SIZE = 6
CONV = 3
out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(CONV)
problem = CudaProblem(
    "1D Conv (Simple)",
    conv_test,
    [a, b],
    out,
    [SIZE, CONV],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)

problem.check()