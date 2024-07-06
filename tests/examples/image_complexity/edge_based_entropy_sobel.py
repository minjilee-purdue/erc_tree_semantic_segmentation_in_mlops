import numpy as np
from skimage.filters import sobel  # edge-based entropy using the Sobel filter from skimage and computes entropy based on the edge information. 
from scipy.stats import entropy

def calculate_entropy(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return entropy(hist)

def calculate_edge_based_entropy_sobel(image):
    edges = sobel(image)
    edges = (edges * 255).astype(np.uint8)
    image_entropy = calculate_entropy(edges)
    return image_entropy

'''
    Note.
    Both Sobel and Canny are used to find edges in images, and they operate on grayscale images (where each pixel has a single intensity value).
    Both methods involve calculating gradients to identify areas of rapid intensity change.
'''