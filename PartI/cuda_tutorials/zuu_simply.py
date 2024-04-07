import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time

# Define the CUDA kernel
#@jit
def complex_function_kernel(output, shape, color_map):
    for x in range(0,8*32*16):
        for y in range (0,8*32*16):
            # Mapping grid position to complex plane, scaling to get a range of values
            z = complex(float(x) / shape[0] * 4.0 - 2.0, float(y) / shape[1] * 4.0 - 2.0)
            # Example complex function
            f_z = z ** 2  # Using a simple complex square function
            # Map the magnitude of f(z) to a color
            magnitude = abs(f_z)
            color_index = int(magnitude * 10 % len(color_map))  # Simple mapping to color index
            output[x, y, 0] = color_map[color_index, 0]
            output[x, y, 1] = color_map[color_index, 1]
            output[x, y, 2] = color_map[color_index, 2]
            output[x, y, 3] = color_map[color_index, 3]

start_time = time.time()

# Colors
colors = np.array([
   [236.0/255, 244.0/255, 214.0/255, 1.0],  # Soft Green
   [154.0/255, 208.0/255, 194.0/255, 1.0],  # Aquamarine
   [45.0/255, 149.0/255, 150.0/255, 1.0],   # Teal
   [38.0/255, 80.0/255, 115.0/255, 1.0],    # Deep Sky Blue
   [34.0/255, 9.0/255, 44.0/255, 1.0],      # Dark Purple
   [135.0/255, 35.0/255, 65.0/255, 1.0],    # Crimson
   [190.0/255, 49.0/255, 68.0/255, 1.0],    # Raspberry
   [240.0/255, 89.0/255, 65.0/255, 1.0],    # Coral
   [7.0/255, 102.0/255, 173.0/255, 1.0],    # Cobalt Blue
   [41.0/255, 173.0/255, 178.0/255, 1.0]    # Turquoise
],dtype=np.float32)

# Copy the output array to the GPU - where our computations will take place

shape=(8*32*16, 8*32*16, 4)
cpu_array = np.zeros(shape, dtype=np.float32)

# Launch the kernel
complex_function_kernel(cpu_array, shape, colors)

end_time = time.time() - start_time
print(f"Execution time: {end_time} seconds")

# Plotting the generated numbers
plt.figure(figsize=(10, 10))
plt.imshow(cpu_array, cmap='hot')
plt.colorbar()
plt.title("Output of the CUDA Kernel")
plt.savefig('zuu.png')