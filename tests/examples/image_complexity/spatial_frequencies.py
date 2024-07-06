import cv2
import numpy as np

def calculate_spatial_frequencies(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform 2D Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
    
    # Calculate spatial frequencies
    spatial_frequencies = np.sum(magnitude_spectrum) / (image.shape[0] * image.shape[1])
    
    # Print spatial frequencies result
    # print(f"Spatial Frequencies: {spatial_frequencies:.4f}")
    
    return spatial_frequencies


'''
Spatial Frequencies measure the average frequency content in an image.
High Spatial Frequencies indicate a complex image with fine details and textures.
Low Spatial Frequencies indicate a simpler image with uniform areas and fewer details.
This metric helps in quantifying the visual complexity of an image based on its frequency content.
'''