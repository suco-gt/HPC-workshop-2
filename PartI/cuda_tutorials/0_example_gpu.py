def sum_of_squares_serial(arr):
    result = 0
    for num in arr:
        result += num * num
    return result

from numba import njit

@njit
def sum_of_squares_numba(arr):
    result = 0
    for num in arr:
        result += num * num
    return result

import numpy as np
import time

# Create a large array of random numbers
arr = np.random.rand(10000000)

# Measure the time taken by the serial function
start_time = time.time()
sum_of_squares_serial(arr)
serial_time = time.time() - start_time

# Measure the time taken by the Numba-accelerated function
start_time = time.time()
sum_of_squares_numba(arr)
numba_time = time.time() - start_time

print(f"Serial execution time: {serial_time} seconds")
print(f"Numba execution time: {numba_time} seconds")
