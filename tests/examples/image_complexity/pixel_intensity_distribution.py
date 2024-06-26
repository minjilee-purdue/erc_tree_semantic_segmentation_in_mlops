import cv2
import numpy as np
from scipy.stats import entropy

def calculate_pixel_intensity_distribution(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram of pixel intensities
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256), density=True)
    
    # Calculate entropy of the histogram
    pixel_entropy = entropy(hist)
    
    # Print pixel intensity distribution (entropy) result
    # print(f"Pixel Intensity Distribution (Entropy): {pixel_entropy:.4f}")
    
    return pixel_entropy


'''
Pixel Intensity Distribution Entropy measures how varied the pixel intensities are in an image.
High entropy signifies a complex image with a lot of details and variations, while low entropy signifies a simpler image with more uniform intensities.
This metric helps in quantifying the visual complexity of an image based on its pixel intensity variations.
'''