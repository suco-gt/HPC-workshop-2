from numba import jit

def sum_of_squares(arr):
    result = 0
    for num in arr:
        result += num * num
    return result

from numba import njit

import numpy as np
import time

# Create a large array of random numbers
arr = np.random.rand(100000000)

# Measure the time taken by the serial function
start_time = time.time()
sum_of_squares(arr)
serial_time = time.time() - start_time

print(f"Execution time: {serial_time} seconds")
