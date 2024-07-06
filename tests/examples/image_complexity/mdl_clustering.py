'''
import os
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from PIL import Image
import cv2

def MDL_Cluster(D, K_max=10):
    best_DL = float('inf')
    best_A = None
    for K in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=0)
        kmeans.fit(D)
        DL = -np.sum(np.log(np.max(kmeans.transform(D), axis=1)))
        if DL < best_DL:
            best_A = kmeans.labels_
            best_DL = DL
    return best_A

def extract_patches(A, m, i, j):
    patch = A[i:i+m, j:j+m].flatten()
    return patch

def Compute_Patch_Signatures(X, m, K_max):
    H, W, _ = X.shape
    A = MDL_Cluster(X.reshape(-1, X.shape[-1]), K_max)
    A = A.reshape(H, W)
    B = Parallel(n_jobs=-1)(delayed(extract_patches)(A, m, i, j) for i in range(H - m + 1) for j in range(W - m + 1))
    B = np.array(B).reshape(H - m + 1, W - m + 1, -1)
    return A, B

def load_image(image_path, scale_factor=0.25):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    return image

if __name__ == "__main__":
    image_folder = '/home/minjilee/Desktop/floridatrip'
    output_folder = '/home/minjilee/Desktop/floridatrip2'
    os.makedirs(output_folder, exist_ok=True)

    scales = [4, 8, 16, 32]
    K_max = 5

    results = []

    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        image = load_image(image_path, scale_factor=0.25)

        _, all_A = Compute_Patch_Signatures(image, scales[0], K_max)

        result = {
            "filename": image_filename,
            "clustering_results": all_A
        }
        results.append(result)

        with open(os.path.join(output_folder, f'mld_results_{image_filename}.txt'), 'w') as f:
            f.write("Clustering Results:\n")
            f.write(str(all_A))

'''

'''
# Visualization
import os
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def MDL_Cluster(D, K_max=10):
    best_DL = float('inf')
    best_A = None
    for K in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=0)
        kmeans.fit(D)
        DL = -np.sum(np.log(np.max(kmeans.transform(D), axis=1)))
        if DL < best_DL:
            best_A = kmeans.labels_
            best_DL = DL
    return best_A

def extract_patches(A, m, i, j):
    patch = A[i:i+m, j:j+m].flatten()
    return patch

def Compute_Patch_Signatures(X, m, K_max):
    H, W, _ = X.shape
    A = MDL_Cluster(X.reshape(-1, X.shape[-1]), K_max)
    A = A.reshape(H, W)
    B = Parallel(n_jobs=-1)(delayed(extract_patches)(A, m, i, j) for i in range(H - m + 1) for j in range(W - m + 1))
    B = np.array(B).reshape(H - m + 1, W - m + 1, -1)
    return A, B

def load_image(image_path, scale_factor=0.25):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    return image

if __name__ == "__main__":
    image_folder = '/home/minjilee/Desktop/floridatrip'
    output_folder = '/home/minjilee/Desktop/floridatrip2'
    os.makedirs(output_folder, exist_ok=True)

    scales = [4, 8, 16, 32]
    K_max = 5

    results = []

    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        image = load_image(image_path, scale_factor=0.25)

        # Compute clustering results
        _, all_A = Compute_Patch_Signatures(image, scales[0], K_max)

        # Reshape all_A if needed for visualization
        # For example, if all_A is (573, 861, 16), reshape it to (573, 861) or (573, 861, 3) if possible
        # Here, we assume all_A needs to be reshaped to visualize it as an image
        if all_A.ndim == 3:
            all_A = np.mean(all_A, axis=-1)  # Take mean over the last dimension to collapse it

        # Visualize clustering results
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        # Clustering result
        plt.subplot(1, 2, 2)
        plt.imshow(all_A, cmap='viridis')  # Use a colormap appropriate for your data
        plt.title('Clustering Results')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Save the visualization if needed
        #plt.savefig(os.path.join(output_folder, f'visualization_{image_filename}.png'))
'''

'''
import os
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import math
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't require a GUI
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import cv2
from skimage.filters import sobel
from scipy.stats import entropy

def MDL_Cluster(D, K_max=10):
    """
    Perform MDL clustering on data matrix D.
    
    Parameters:
    - D: Input data matrix (flattened to 2D).
    - K_max: Maximum number of clusters to consider.
    
    Returns:
    - best_A: Best clustering labels.
    """
    best_DL = float('inf')
    best_A = None
    for K in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=0)
        kmeans.fit(D)
        labels = kmeans.labels_
        DL = -np.sum(np.log(np.max(kmeans.transform(D), axis=1)))
        if DL < best_DL:
            best_A = labels
            best_DL = DL
    return best_A

def Signatures_Entropy(S):
    """
    Calculate entropy of signature data S.
    
    Parameters:
    - S: Signature data.
    
    Returns:
    - entropy: Entropy value.
    """
    bin_counts = Counter(map(tuple, S))
    total = sum(bin_counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in bin_counts.values())
    return entropy

def extract_patches(A, m, i, j):
    """
    Extract patches from matrix A.
    
    Parameters:
    - A: Input matrix.
    - m: Patch size.
    - i, j: Indices to extract patch.
    
    Returns:
    - signature_vector: Flattened signature vector of the patch.
    """
    patch = A[i:i+m, j:j+m].flatten()
    signature = Counter(patch)
    signature_vector = [signature.get(k, 0) for k in range(np.max(A) + 1)]
    return signature_vector

def Compute_Patch_Signatures(X, m, K_max):
    """
    Compute patch signatures for input matrix X.
    
    Parameters:
    - X: Input matrix.
    - m: Patch size.
    - K_max: Maximum number of clusters.
    
    Returns:
    - A: Clustering labels.
    - B: Patch signatures.
    """
    H, W, _ = X.shape
    A = MDL_Cluster(X.reshape(-1, X.shape[-1]), K_max)
    A = A.reshape(H, W)
    B = Parallel(n_jobs=-1)(delayed(extract_patches)(A, m, i, j) for i in range(H - m + 1) for j in range(W - m + 1))
    B = np.array(B).reshape(H - m + 1, W - m + 1, -1)
    return A, B

def Complexity(X, scales, K_max):
    """
    Compute complexity measures for input matrix X across multiple scales.
    
    Parameters:
    - X: Input matrix.
    - scales: List of scales (patch sizes).
    - K_max: Maximum number of clusters.
    
    Returns:
    - total_complexity: Total complexity measure.
    - all_A: List of clustering labels for each scale.
    - signature_entropies: List of patch signature entropies for each scale.
    """
    total_complexity = 0
    all_A = []
    signature_entropies = []
    for m in scales:
        A, X = Compute_Patch_Signatures(X, m, K_max)
        sig_entropy = Signatures_Entropy(X.reshape(-1, X.shape[-1]))
        total_complexity += sig_entropy
        signature_entropies.append(sig_entropy)
        all_A.append(A)
    return total_complexity, all_A, signature_entropies

def load_image(image_path, scale_factor=0.25):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    return image

def visualize_clustering(image, all_A, scales, output_folder, filename):
    """
    Visualize clustering results and save as .png files.
    
    Parameters:
    - image: Original image.
    - all_A: List of clustering labels for each scale.
    - scales: List of scales (patch sizes).
    - output_folder: Folder to save output images.
    - filename: Filename prefix for output images.
    """
    fig, axes = plt.subplots(1, len(scales) + 1, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i, (A, m) in enumerate(zip(all_A, scales)):
        axes[i + 1].imshow(A, cmap='viridis')
        axes[i + 1].set_title(f'Clustering at Scale {m}')
        axes[i + 1].axis('off')
    
    plt.savefig(os.path.join(output_folder, f'{filename}_clustering.png'))
    plt.close('all')  # Close the figure explicitly


if __name__ == "__main__":
    image_folder = '/home/minjilee/Desktop/floridatrip'
    output_folder = '/home/minjilee/Desktop/floridatrip/mdl_cluster'
    scales = [4, 8, 16, 32]
    K_max = 5

    os.makedirs(output_folder, exist_ok=True)

    results = []

    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        image = load_image(image_path, scale_factor=0.25)

        complexity, all_A, signature_entropies = Complexity(image, scales, K_max)

        result = {
            "filename": image_filename,
            "signature_entropies": signature_entropies,
            "total_complexity": f"{complexity:.4f}"
        }
        results.append(result)

        # Visualize and save clustering results
        visualize_clustering(image, all_A, scales, output_folder, os.path.splitext(image_filename)[0])

        # Save patch entropy values to a text file
        with open(os.path.join(output_folder, f'results_{os.path.splitext(image_filename)[0]}.txt'), 'w') as f:
            for scale, entropy in zip(scales, signature_entropies):
                f.write(f"Patch Entropy ({scale}): {entropy:.4f}\n")
                f.write(f"Total complexity (MDL Clustering): {result['total_complexity']}\n")

'''



import os
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import math
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import cv2
from scipy.stats import entropy

def MDL_Cluster(D, K_max=10):
    best_DL = float('inf')
    best_A = None
    for K in range(1, K_max + 1):
        kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=0)
        kmeans.fit(D)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        DL = -np.sum(np.log(np.max(kmeans.transform(D), axis=1)))
        if DL < best_DL:
            best_A = labels
            best_DL = DL
    return best_A

def Signatures_Entropy(S):
    bin_counts = Counter(map(tuple, S))
    total = sum(bin_counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in bin_counts.values())
    return entropy

def extract_patches(A, m, i, j):
    patch = A[i:i+m, j:j+m].flatten()
    signature = Counter(patch)
    signature_vector = [signature.get(k, 0) for k in range(np.max(A) + 1)]
    return signature_vector

def Compute_Patch_Signatures(X, m, K_max):
    H, W, _ = X.shape
    A = MDL_Cluster(X.reshape(-1, X.shape[-1]), K_max)
    A = A.reshape(H, W)
    B = Parallel(n_jobs=-1)(delayed(extract_patches)(A, m, i, j) for i in range(H - m + 1) for j in range(W - m + 1))
    B = np.array(B).reshape(H - m + 1, W - m + 1, -1)
    return A, B

def Complexity(X, scales, K_max):
    total_complexity = 0
    all_A = []
    signature_entropies = []
    for m in scales:
        A, X = Compute_Patch_Signatures(X, m, K_max)
        sig_entropy = Signatures_Entropy(X.reshape(-1, X.shape[-1]))
        total_complexity += sig_entropy
        signature_entropies.append(sig_entropy)
        all_A.append(A)
    return total_complexity, all_A, signature_entropies

def load_image(image_path, scale_factor=0.25):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    return image

def visualize_clustering(image, all_A, scales, output_folder, filename):
    fig, axes = plt.subplots(1, len(scales) + 1, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i, (A, m) in enumerate(zip(all_A, scales)):
        axes[i + 1].imshow(A, cmap='viridis')
        axes[i + 1].set_title(f'Clustering at Scale {m}')
        axes[i + 1].axis('off')
    
    plt.savefig(os.path.join(output_folder, f'{filename}_clustering.png'))
    plt.close('all')

if __name__ == "__main__":
    image_folder = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/raw/tree'
    output_folder = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/complexity/mdl_cluster'
    
    os.makedirs(output_folder, exist_ok=True)

    scales = [4, 8, 16, 32]
    K_max = 5

    results = []

    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        if os.path.isfile(image_path) and image_filename.lower().endswith('.jpg'):
            image = load_image(image_path, scale_factor=0.25)
            try:
                complexity, all_A, signature_entropies = Complexity(image, scales, K_max)
                result = {
                    "filename": image_filename,
                    "total_complexity": f"{complexity:.4f}",
                    "signature_entropies": {scale: f"{entropy:.4f}" for scale, entropy in zip(scales, signature_entropies)}
                }
                results.append(result)

                result_file_path = os.path.join(output_folder, f'results_{os.path.splitext(image_filename)[0]}.txt')
                with open(result_file_path, 'w') as f:
                    f.write(f"Total complexity (MDL Clustering): {result['total_complexity']}\n")
                    for scale, sig_entropy in result["signature_entropies"].items():
                        f.write(f"Patch-based Signature Entropy at scale {scale}: {sig_entropy}\n")

                visualize_clustering(image, all_A, scales, output_folder, os.path.splitext(image_filename)[0])

            except Exception as e:
                print(f"Error processing {image_filename}: {e}")

    print(f"All metrics saved to {output_folder}.")
