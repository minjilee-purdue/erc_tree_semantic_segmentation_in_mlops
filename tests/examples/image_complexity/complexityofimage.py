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


def calculate_entropy(image):

    hist, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return entropy(hist)

def edge_based_entropy(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = sobel(image)
    edges = (edges * 255).astype(np.uint8)
    image_entropy = calculate_entropy(edges)
    return image_entropy

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
    plt.close('all')  # Close the figure explicitly


if __name__ == "__main__":
    image_folder = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/raw/tree'
    output_folder = '/home/minjilee/Desktop/seasonaldata/feb/evaluation/complexity'
    os.makedirs(output_folder, exist_ok=True)

    scales = [4, 8, 16, 32]
    K_max = 5

    results = []

    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        image = load_image(image_path, scale_factor=0.25)

        edge_entropy = edge_based_entropy(image_path)
        complexity, all_A, signature_entropies = Complexity(image, scales, K_max)

        result = {
            "filename": image_filename,
            "edge_entropy": f"{edge_entropy:.4f}",
            "total_complexity": f"{complexity:.4f}",
            "signature_entropies": {scale: f"{entropy:.4f}" for scale, entropy in zip(scales, signature_entropies)}
        }
        results.append(result)

        with open(os.path.join(output_folder, f'results_{image_filename}.txt'), 'w') as f:
            f.write(f"Edge-based Entropy: {result['edge_entropy']}\n")
            f.write(f"Total complexity (MDL Clustering): {result['total_complexity']}\n")
            for scale, sig_entropy in result["signature_entropies"].items():
                f.write(f"Patch-based Signature Entropy at scale {scale}: {sig_entropy}\n")

        visualize_clustering(image, all_A, scales, output_folder, os.path.splitext(image_filename)[0])