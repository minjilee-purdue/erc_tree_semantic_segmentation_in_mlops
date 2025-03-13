'''
example of the torch Dataset and DataLoader
'''

from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels, bbox):
        self.data = data
        self.labels = labels
        self.bbox = bbox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        bbox = self.bbox[idx]
        return x, y, bbox

# create just sample and random data
data = torch.randn(100, 3, 224, 224)  # e.g., 224x224 size RGB image 100 data
labels = torch.randint(0, 10, (100,))  # e.g., label for 100 data (0-9 integer)
bbox = torch.randint(0, 224, (100, 4))  # e.g., bounding box for 100 data (x_min, y_min, x_max, y_max)

# create dataset and dataloader
dataset = CustomDataset(data, labels, bbox)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


'''
Batch size refers to the number of data samples that are processed at once when training machine learning and deep learning models. Instead of processing the entire dataset at once, the model training process divides the data into batches. Batch size has a significant impact on the performance and speed of training, and is characterized by the following features

    Small batch size:
        Less memory usage.
        Weights can be updated more frequently, allowing the model to converge faster.
        Convergence speed can be unstable due to high noise.

    Large batch size:
        Memory usage is high.
        May have more stable convergence with less frequent weight updates.
        Can result in longer computation times.

The Batch Loader is responsible for dividing the dataset into batches and feeding them to the model. In PyTorch, you can load data in batches using the DataLoader class.
The DataLoader extracts a batch-sized amount of data from a given dataset and makes it available for model training.

# Iterate through the dataloader
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f'Batch {batch_idx + 1}:')
    print(f'  Data shape: {data.shape}')
    print(f'  Labels shape: {labels.shape}')
  
The first 26 batches have a shape of [4, 3, 224, 224] for the data and [4] for the labels.
The 27th batch has a shape of [2, 3, 224, 224] for the data and [2] for the labels because there are only 2 samples remaining.

'''
for batch_data, batch_labels, batch_bbox in dataloader:
    print(batch_data.size(), batch_labels.size(), batch_bbox.size())
