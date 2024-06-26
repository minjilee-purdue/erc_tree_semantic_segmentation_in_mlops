'''

import os
import pandas as pd

# Folder containing your .txt files
folder_path = '/home/minjilee/Desktop/seasonaldata/feb/all_complexity/results' 
output_file = '/home/minjilee/Desktop/seasonaldata/feb/all_complexity/entropy_data.xlsx'  # Change this to the desired output path

# Initialize an empty list to hold data
data = []

# Iterate over all .txt files in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()
            # Extract the data from the file
            edge_entropy = float(lines[0].split(': ')[1].strip())
            total_complexity = float(lines[1].split(': ')[1].strip())
            patch_entropy_4 = float(lines[2].split(': ')[1].strip())
            patch_entropy_8 = float(lines[3].split(': ')[1].strip())
            patch_entropy_16 = float(lines[4].split(': ')[1].strip())
            patch_entropy_32 = float(lines[5].split(': ')[1].strip())

            # Append the data to the list
            data.append({
                'Filename': filename,
                'Edge Entropy': edge_entropy,
                'Total complexity (MDL Clustering)': total_complexity,
                'Patch Entropy (4)': patch_entropy_4,
                'Patch Entropy (8)': patch_entropy_8,
                'Patch Entropy (16)': patch_entropy_16,
                'Patch Entropy (32)': patch_entropy_32
            })

# Create a DataFrame from the list of data
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel(output_file, index=False)

print(f'Data has been successfully saved to {output_file}')

'''



import os
import pandas as pd

# Define folder paths for each type of analysis
folder_paths = {
    "edge_based_entropy_sobel": "/home/minjilee/Desktop/seasonaldata/feb/all_complexity/results/edge_based_entropy_sobel",
    "edge_density_canny": "/home/minjilee/Desktop/seasonaldata/feb/all_complexity/results/edge_density_canny",
    "mdl_total": "/home/minjilee/Desktop/seasonaldata/feb/all_complexity/results/mdl_total",
    "pixel_intensity_distribution": "/home/minjilee/Desktop/seasonaldata/feb/all_complexity/results/pixel_intensity_distribution",
    "spatial_frequencies": "/home/minjilee/Desktop/seasonaldata/feb/all_complexity/results/spatial_frequencies"
}

# Initialize an empty dictionary to hold data for each image
image_data = {}

# Function to extract values from each directory
def extract_values_from_directory(directory_path, image_number):
    data = {}
    for filename in os.listdir(directory_path):
        if filename.startswith(f"image_{image_number}_") or filename.startswith(f"results_image_{image_number}"):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith(".txt"):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('Edge-based Entropy Sobel:'):
                            data['Edge Entropy'] = float(line.split(':')[1].strip())
                        elif line.startswith('Edge Density Canny:'):
                            data['Edge Density'] = float(line.split(':')[1].strip())
                        elif line.startswith('Pixel Intensity Distribution (Entropy):'):
                            data['Pixel Intensity Distribution'] = float(line.split(':')[1].strip())
                        elif line.startswith('Spatial Frequencies:'):
                            data['Spatial Frequencies'] = float(line.split(':')[1].strip())
                        elif line.startswith('Total complexity (MDL Clustering):'):
                            data['Total Complexity'] = float(line.split(':')[1].strip())
                        elif line.startswith('Patch-based Signature Entropy at scale 4:'):
                            data['Patch Entropy (4)'] = float(line.split(':')[1].strip())
                        elif line.startswith('Patch-based Signature Entropy at scale 8:'):
                            data['Patch Entropy (8)'] = float(line.split(':')[1].strip())
                        elif line.startswith('Patch-based Signature Entropy at scale 16:'):
                            data['Patch Entropy (16)'] = float(line.split(':')[1].strip())
                        elif line.startswith('Patch-based Signature Entropy at scale 32:'):
                            data['Patch Entropy (32)'] = float(line.split(':')[1].strip())
    return data

# Iterate over each analysis type and extract the values
for analysis_type, folder_path in folder_paths.items():
    for filename in os.listdir(folder_path):
        if filename.startswith("image_") or filename.startswith("results_image_"):
            image_number = ''.join(filter(str.isdigit, filename))  # Extract numeric part from filename
            image_name = f"image_{image_number}"
            if image_name not in image_data:
                image_data[image_name] = extract_values_from_directory(folder_path, image_number)
            else:
                # Update existing data (if additional files exist for the same image)
                image_data[image_name].update(extract_values_from_directory(folder_path, image_number))

# Sort the keys of image_data dictionary
sorted_keys = sorted(image_data.keys(), key=lambda x: int(x.split('_')[1]))

# Define the desired order of columns
column_order = ['Edge Entropy', 'Edge Density', 'Pixel Intensity Distribution', 'Spatial Frequencies', 'Total Complexity',
                'Patch Entropy (4)', 'Patch Entropy (8)', 'Patch Entropy (16)', 'Patch Entropy (32)']

# Create a DataFrame from the sorted dictionary with the specified column order
df = pd.DataFrame.from_dict({key: image_data[key] for key in sorted_keys}, orient='index', columns=column_order)

# Save the DataFrame to an Excel file
output_file = '/home/minjilee/Desktop/seasonaldata/feb/all_complexity/total_image_analysis.xlsx'
df.to_excel(output_file, index=True)
print(f'Data has been successfully saved to {output_file}')