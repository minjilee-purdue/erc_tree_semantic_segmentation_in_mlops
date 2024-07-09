# notebook_utils.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(data, title="Histogram"):
    """Plot a histogram of data."""
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, alpha=0.75)
    plt.title(title)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def display_dataframe(df, num_rows=5):
    """Display the first few rows of a DataFrame."""
    display(df.head(num_rows))

def save_figure(fig, filename):
    """Save a matplotlib figure to a file."""
    fig.savefig(filename)
    print(f"Figure saved as {filename}")
