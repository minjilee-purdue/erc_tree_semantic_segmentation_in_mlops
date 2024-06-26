'''
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Define the directory where the txt files are stored
directory = "/home/minjilee/Desktop/seasonaldata/feb/complexity/results"

# Initialize lists to store data
edge_entropy = []
total_complexity = []
patch_entropy_4 = []
patch_entropy_8 = []
patch_entropy_16 = []
patch_entropy_32 = []

# Iterate over each txt file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            edge_entropy.append(float(lines[0].split(":")[1].strip()))
            total_complexity.append(float(lines[1].split(":")[1].strip()))
            patch_entropy_4.append(float(lines[2].split(":")[1].strip()))
            patch_entropy_8.append(float(lines[3].split(":")[1].strip()))
            patch_entropy_16.append(float(lines[4].split(":")[1].strip()))
            patch_entropy_32.append(float(lines[5].split(":")[1].strip()))

# Convert lists to numpy arrays for easier manipulation
edge_entropy = np.array(edge_entropy)
total_complexity = np.array(total_complexity)
patch_entropy_4 = np.array(patch_entropy_4)
patch_entropy_8 = np.array(patch_entropy_8)
patch_entropy_16 = np.array(patch_entropy_16)
patch_entropy_32 = np.array(patch_entropy_32)

# Calculate the average patch-based entropy across scales
avg_patch_entropy = (patch_entropy_4 + patch_entropy_8 + patch_entropy_16 + patch_entropy_32) / 4

# Calculate the correlation coefficients
correlation_coefficient_edge = np.corrcoef(edge_entropy, total_complexity)[0, 1]
correlation_coefficient_patch = np.corrcoef(avg_patch_entropy, total_complexity)[0, 1]

# Perform linear regression for edge entropy and patch-based entropy
slope_edge, intercept_edge, _, _, _ = linregress(edge_entropy, total_complexity)
line_of_best_fit_edge = slope_edge * edge_entropy + intercept_edge

slope_patch, intercept_patch, _, _, _ = linregress(avg_patch_entropy, total_complexity)
line_of_best_fit_patch = slope_patch * avg_patch_entropy + intercept_patch

# Plot the scatter plot
plt.figure(figsize=(12, 6))

# Plot for edge entropy
plt.subplot(1, 2, 1)
plt.scatter(edge_entropy, total_complexity, color="skyblue", label="Data Points")
plt.plot(edge_entropy, line_of_best_fit_edge, color="red", label="Trend Line")
plt.xlabel("Edge Entropy")
plt.ylabel("Total Complexity")
plt.title("Edge Entropy vs Total Complexity")
plt.legend()
plt.grid(True)

# Annotate the plot with the correlation coefficient for edge entropy
plt.text(0.05, 0.95, f"Correlation Coefficient: {correlation_coefficient_edge:.2f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

# Plot for patch-based entropy
plt.subplot(1, 2, 2)
plt.scatter(avg_patch_entropy, total_complexity, color="skyblue", label="Data Points")
plt.plot(avg_patch_entropy, line_of_best_fit_patch, color="red", label="Trend Line")
plt.xlabel("Average Patch Entropy")
plt.ylabel("Total Complexity")
plt.title("Average Patch Entropy vs Total Complexity")
plt.legend()
plt.grid(True)

# Annotate the plot with the correlation coefficient for patch-based entropy
plt.text(0.05, 0.95, f"Correlation Coefficient: {correlation_coefficient_patch:.2f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()

'''

'''
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Define the directory where the txt files are stored
directory = "/home/minjilee/Desktop/seasonaldata/feb/all_complexity/entropy_data.xlsx"

# Initialize lists to store data
edge_entropy = []
avg_patch_entropy = []
total_complexity = []

# Iterate over each txt file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
            edge_entropy.append(float(lines[0].split(":")[1].strip()))
            patch_entropy_4 = float(lines[2].split(":")[1].strip())
            patch_entropy_8 = float(lines[3].split(":")[1].strip())
            patch_entropy_16 = float(lines[4].split(":")[1].strip())
            patch_entropy_32 = float(lines[5].split(":")[1].strip())
            avg_patch_entropy.append((patch_entropy_4 + patch_entropy_8 + patch_entropy_16 + patch_entropy_32) / 4)
            total_complexity.append(float(lines[1].split(":")[1].strip()))

# Convert lists to numpy arrays for easier manipulation
edge_entropy = np.array(edge_entropy).reshape(-1, 1)
avg_patch_entropy = np.array(avg_patch_entropy).reshape(-1, 1)
total_complexity = np.array(total_complexity)

# Perform multiple linear regression
X = np.hstack((edge_entropy, avg_patch_entropy))
regressor = LinearRegression()
regressor.fit(X, total_complexity)

# Predict total complexity using the regression model
predicted_complexity = regressor.predict(X)

# Plot the scatter plot
fig = plt.figure(figsize=(12, 6))

# Plot for edge entropy
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(edge_entropy, avg_patch_entropy, total_complexity, color='skyblue')
ax1.plot_trisurf(edge_entropy.flatten(), avg_patch_entropy.flatten(), predicted_complexity, color='red', alpha=0.5)
ax1.set_xlabel('Edge Entropy')
ax1.set_ylabel('Average Patch Entropy')
ax1.set_zlabel('Total Complexity')
ax1.set_title('Multiple Linear Regression')

# Annotate the plot with the equation of the plane
equation = f'Complexity = {regressor.intercept_:.2f} + {regressor.coef_[0]:.2f} * Edge Entropy + {regressor.coef_[1]:.2f} * Avg Patch Entropy'
ax1.text2D(0.05, 0.95, equation, transform=ax1.transAxes, fontsize=10)

# Plot for average patch entropy
ax2 = fig.add_subplot(122)
ax2.scatter(total_complexity, predicted_complexity, color='skyblue')
ax2.plot(total_complexity, total_complexity, color='red', linestyle='--')
ax2.set_xlabel('Actual Total Complexity')
ax2.set_ylabel('Predicted Total Complexity')
ax2.set_title('Actual vs Predicted Total Complexity')

plt.tight_layout()
plt.show()
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the provided Excel file
file_path = '/home/minjilee/Desktop/seasonaldata/feb/all_complexity/entropy_data.xlsx'
df = pd.read_excel(file_path)

# Extract the relevant data for edge_entropy, total_complexity, and patch_entropies
edge_entropy = df['Edge Entropy'].values
total_complexity = df['Total Complexity'].values
patch_entropies = df[['Patch Entropy (4)', 
                      'Patch Entropy (8)',
                      'Patch Entropy (16)',
                      'Patch Entropy (32)']].values

# Combine all variables into one array
data = np.hstack((edge_entropy.reshape(-1, 1), total_complexity.reshape(-1, 1), patch_entropies))

# Compute the correlation matrix
correlation_matrix = np.corrcoef(data, rowvar=False)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=['Edge Entropy', 'Total Complexity', 'Patch Entropy (4)', 'Patch Entropy (8)', 'Patch Entropy (16)', 'Patch Entropy (32)'],
            yticklabels=['Edge Entropy', 'Total Complexity', 'Patch Entropy (4)', 'Patch Entropy (8)', 'Patch Entropy (16)', 'Patch Entropy (32)'])
plt.title('Correlation Matrix')
plt.show()