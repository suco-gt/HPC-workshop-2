import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import time
from save_image import save_image

# Define the CUDA kernel
@cuda.jit
def generate_one_tile(output, color_map):
    tile_x, tile_y = cuda.grid(2)
    for xi in range(16):
        for yi in range(16):
            x,y=tile_x*16+xi, tile_y*16+yi
            
            # Color the pixel with respect to the value of the complex function at the pixels coordinate
            z = complex(float(x) / output.shape[0] * 2.0 - 1.0, float(y) / output.shape[1] * 2.0 - 1.0)
            f_z = z ** 1.5 ** (complex(-1,-1)) if z != 0 else 0
            magnitude = abs(f_z)
            color_index = int(magnitude * 10 % len(color_map))
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

colors_gpu = cuda.to_device(colors)

# Copy the output array to the GPU - where our computations will take place
shape=(4096, 4096, 4) # = 32*8*16, 32*8*16, 4
image_gpu = cuda.device_array(shape,dtype=np.float32)

# Grid and block dimensions
blocks_per_grid = (32, 32)
threads_per_block = (8, 8)

# Launch the kernel
generate_one_tile[blocks_per_grid, threads_per_block](image_gpu, colors_gpu)

# Copy the result of our computations back to the CPU - to save as a PNG file
cpu_array = image_gpu.copy_to_host()

end_time = time.time() - start_time
print(f"Execution time: {end_time} seconds")

save_image(cpu_array, "output.png")