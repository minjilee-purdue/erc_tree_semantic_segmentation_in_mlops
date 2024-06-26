'''
All functions and modules are included except mdl_clustering.py
| edge_density_canny.py, edge_based_entropy_sobel.py, pixel_intensity_distribution.py, and spatial_frequencies.py
'''

import os
import cv2
from edge_density_canny import calculate_edge_density
from edge_based_entropy_sobel import calculate_edge_based_entropy_sobel
from pixel_intensity_distribution import calculate_pixel_intensity_distribution
from spatial_frequencies import calculate_spatial_frequencies


if __name__ == "__main__":
    # Folder containing images
    image_folder = '/home/minjilee/Desktop/seasonaldata/feb/training/raw/tree'
    
    # Output folder for edge density results
    output_folder_edge_density_canny = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/complexity/edge_density_canny'
    output_folder_edge_based_entropy_sobel = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/complexity/edge_based_entropy_sobel'
    output_folder_pixel_intensity_distribution = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/complexity/pixel_intensity_distribution'
    output_folder_spatial_frequencies = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/complexity/spatial_frequencies'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder_edge_density_canny, exist_ok=True)
    os.makedirs(output_folder_edge_based_entropy_sobel, exist_ok=True)
    os.makedirs(output_folder_pixel_intensity_distribution, exist_ok=True)
    os.makedirs(output_folder_spatial_frequencies, exist_ok=True)

    # Iterate over each image file in the folder
    for image_filename in sorted(os.listdir(image_folder)):
        if image_filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_filename)
            
            # Load image
            image = cv2.imread(image_path)
            
            try:
                # Calculate edge density
                edge_density_canny = calculate_edge_density(image)
                
                # Save edge density to text file in the specific folder
                edge_density_file = os.path.join(output_folder_edge_density_canny, f"{os.path.splitext(image_filename)[0]}_edge_density_canny.txt")
                with open(edge_density_file, 'w') as f:
                    f.write(f"Edge Density Canny: {edge_density_canny:.4f}\n")
                # print(f"Saved Edge Density Canny for {image_filename} to {edge_density_file}")
                
                # Calculate edge-based entropy using Sobel
                edge_entropy_sobel = calculate_edge_based_entropy_sobel(image)
            
                # Save edge-based entropy using Sobel to text file in the specific folder
                edge_entropy_sobel_file = os.path.join(output_folder_edge_based_entropy_sobel, f"{os.path.splitext(image_filename)[0]}_edge_based_entropy_sobel.txt")
                with open(edge_entropy_sobel_file, 'w') as f:
                    f.write(f"Edge-based Entropy Sobel: {edge_entropy_sobel:.4f}\n")
                # print(f"Edge-based Entropy Sobel for {image_filename} to {edge_entropy_sobel_file}")

                # Calculate pixel intensity distribution (entropy)
                pixel_entropy = calculate_pixel_intensity_distribution(image)

                # Save pixel intensity distribution to text file in the specific folder
                pixel_entropy_file = os.path.join(output_folder_pixel_intensity_distribution, f"{os.path.splitext(image_filename)[0]}_pixel_intensity_distribution.txt")
                with open(pixel_entropy_file, 'w') as f:
                    f.write(f"Pixel Intensity Distribution (Entropy): {pixel_entropy:.4f}\n")
                # print(f"Saved Pixel Intensity Distribution (Entropy) for {image_filename} to {pixel_entropy_file}")

                # Calculate spatial frequencies
                spatial_frequencies = calculate_spatial_frequencies(image)
            
                # Save spatial frequencies to text file in the specific folder
                spatial_frequencies_file = os.path.join(output_folder_spatial_frequencies, f"{os.path.splitext(image_filename)[0]}_spatial_frequencies.txt")
                with open(spatial_frequencies_file, 'w') as f:
                    f.write(f"Spatial Frequencies: {spatial_frequencies:.4f}\n")
                # print(f"Saved Spatial Frequencies for {image_filename} to {spatial_frequencies_file}")
            
            
            except Exception as e:
                print(f"Error processing {image_filename}: {e}")
                continue
                
    print(f"All metrics saved to respective folders.")
