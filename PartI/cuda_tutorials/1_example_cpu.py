import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
from save_image import save_image

def generate_image(output, color_map):
    """
    Populates the output array with an image representing the graph of a complex function
    """
    for x in range(4096):
        for y in range (4096):
            # Color the pixel with respect to the value of the complex function at the pixels coordinate
            z = complex(float(x) / output.shape[0] * 2.0 - 1.0, float(y) / output.shape[1] * 2.0 - 1.0)
            f_z =  z ** 1.5 ** (complex(-1,-1)) if z != 0 else 0
            magnitude = abs(f_z)
            color_index = int(magnitude * 10 % len(color_map))
            output[x, y] = color_map[color_index]

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

shape=(4096, 4096, 4)
image = np.zeros(shape, dtype=np.float32)

# Launch the kernel
generate_image(image, colors)

end_time = time.time() - start_time
print(f"Execution time: {end_time} seconds")

save_image(image,"output.png")