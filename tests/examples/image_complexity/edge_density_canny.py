import cv2
import numpy as np

def calculate_edge_density(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate edge density
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
    
    return edge_density


'''
    Note.
    Both Sobel and Canny are used to find edges in images, and they operate on grayscale images (where each pixel has a single intensity value).
    Both methods involve calculating gradients to identify areas of rapid intensity change.
'''

'''
The input image is smoothed using a 5x5 Gaussian filter to reduce noise.

Intensity Gradient Calculation:
    The smoothed image is filtered with Sobel kernels in both horizontal and vertical directions.
    This yields the first derivatives in the horizontal (Gx) and vertical (Gy) directions.
    The edge gradient magnitude and direction are computed for each pixel:
    Edge_Gradient(G) = sqrt(Gx^2 + Gy^2)
    Angle (θ) = atan(Gy / Gx)
'''