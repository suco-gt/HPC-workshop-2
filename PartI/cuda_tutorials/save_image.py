from PIL import Image
import numpy as np

def save_image(image,name):
    # Normalize image to the range of 0-255 if it's not already
    image_normalized = (image - image.min()) / (image.max() - image.min()) * 255
    image_normalized = image_normalized.astype(np.uint8)  # Convert to unsigned byte
    # Create an image from the array
    img = Image.fromarray(image_normalized, 'RGBA')
    # Save the image
    img.save(name)