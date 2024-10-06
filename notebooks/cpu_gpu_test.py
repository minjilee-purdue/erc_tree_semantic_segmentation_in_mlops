import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(36864, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and move to the selected device
model = SimpleCNN().to(device)

# Define training parameters and prepare the data
batch_size, learning_rate, epochs = 64, 0.01, 5
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/minjilee/data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Function to train the model on a given device
def train_model(device_name, device):
    model.to(device)
    model.train()
    epoch_losses = []
    print(f"Training on {device_name}...")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'{device_name} Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
        epoch_losses.append(epoch_loss / len(train_loader))  # Average loss for the epoch
    total_time = time.time() - start_time
    print(f"Training Time on {device_name}: {total_time:.2f} seconds")
    return epoch_losses, total_time

# Train on CPU and GPU, capturing losses and times
cpu_losses, cpu_training_time = train_model("CPU", torch.device("cpu"))
gpu_losses, gpu_training_time = train_model("GPU", torch.device("cuda"))

# Visualization 1: Training Loss over Epochs for CPU and GPU
plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), cpu_losses, label='CPU Training Loss', marker='o', linestyle='-')
plt.plot(range(1, epochs + 1), gpu_losses, label='GPU Training Loss', marker='s', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs for CPU and GPU')
plt.legend()
plt.grid()
plt.show()

# Visualization 2: Training Time Comparison
devices = ['CPU', 'GPU']
training_times = [cpu_training_time, gpu_training_time]
plt.figure(figsize=(8, 6))
plt.bar(devices, training_times, color=['blue', 'green'])
plt.xlabel('Device')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison: CPU vs GPU')
for i, time in enumerate(training_times):
    plt.text(i, time + 1, f'{time:.2f}s', ha='center', va='bottom', fontsize=12)  # Display time on top of each bar
plt.show()

# Visualization 3: Training Speed (Samples per Second)
cpu_speed = len(train_loader.dataset) / cpu_training_time
gpu_speed = len(train_loader.dataset) / gpu_training_time
speeds = [cpu_speed, gpu_speed]
plt.figure(figsize=(8, 6))
plt.bar(devices, speeds, color=['orange', 'purple'])
plt.xlabel('Device')
plt.ylabel('Samples Processed Per Second')
plt.title('Training Speed Comparison: CPU vs GPU')
for i, speed in enumerate(speeds):
    plt.text(i, speed + 10, f'{speed:.2f} samples/s', ha='center', va='bottom', fontsize=12)
plt.show()

# Visualization 4: Combined Training Loss and Time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Subplot 1: Training Loss
ax1.plot(range(1, epochs + 1), cpu_losses, label='CPU Training Loss', marker='o', linestyle='-')
ax1.plot(range(1, epochs + 1), gpu_losses, label='GPU Training Loss', marker='s', linestyle='--')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss over Epochs')
ax1.legend()
ax1.grid()
# Subplot 2: Training Time Bar Chart
ax2.bar(devices, training_times, color=['blue', 'green'])
ax2.set_xlabel('Device')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time Comparison')
ax2.set_ylim(0, max(training_times) * 1.2)
for i, time in enumerate(training_times):
    ax2.text(i, time + 1, f'{time:.2f}s', ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()

# Visualization 5: Cumulative Training Loss
cpu_cumulative_loss = np.cumsum(cpu_losses)
gpu_cumulative_loss = np.cumsum(gpu_losses)
plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), cpu_cumulative_loss, label='CPU Cumulative Loss', marker='o', linestyle='-')
plt.plot(range(1, epochs + 1), gpu_cumulative_loss, label='GPU Cumulative Loss', marker='s', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Cumulative Loss')
plt.title('Cumulative Training Loss over Epochs')
plt.legend()
plt.grid()
plt.show()

# Visualization 6: Epoch Duration Comparison
cpu_epoch_durations = np.linspace(cpu_training_time / epochs, cpu_training_time, epochs)
gpu_epoch_durations = np.linspace(gpu_training_time / epochs, gpu_training_time, epochs)
plt.figure(figsize=(12, 5))
plt.plot(range(1, epochs + 1), cpu_epoch_durations, label='CPU Epoch Duration', marker='o', linestyle='-')
plt.plot(range(1, epochs + 1), gpu_epoch_durations, label='GPU Epoch Duration', marker='s', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Duration (seconds)')
plt.title('Epoch Duration Comparison: CPU vs GPU')
plt.legend()
plt.grid()
plt.show()



'''
Training on CPU...
CPU Train Epoch: 1 [0/60000] Loss: 2.311279
CPU Train Epoch: 1 [6400/60000] Loss: 1.024557
CPU Train Epoch: 1 [12800/60000] Loss: 0.428977
CPU Train Epoch: 1 [19200/60000] Loss: 0.344229
CPU Train Epoch: 1 [25600/60000] Loss: 0.470916
CPU Train Epoch: 1 [32000/60000] Loss: 0.465087
CPU Train Epoch: 1 [38400/60000] Loss: 0.263331
CPU Train Epoch: 1 [44800/60000] Loss: 0.368465
CPU Train Epoch: 1 [51200/60000] Loss: 0.224810
CPU Train Epoch: 1 [57600/60000] Loss: 0.189788
CPU Train Epoch: 2 [0/60000] Loss: 0.197008
CPU Train Epoch: 2 [6400/60000] Loss: 0.328317
CPU Train Epoch: 2 [12800/60000] Loss: 0.159225
CPU Train Epoch: 2 [19200/60000] Loss: 0.267062
CPU Train Epoch: 2 [25600/60000] Loss: 0.254875
CPU Train Epoch: 2 [32000/60000] Loss: 0.185207
CPU Train Epoch: 2 [38400/60000] Loss: 0.128571
CPU Train Epoch: 2 [44800/60000] Loss: 0.168866
CPU Train Epoch: 2 [51200/60000] Loss: 0.223577
CPU Train Epoch: 2 [57600/60000] Loss: 0.191410
CPU Train Epoch: 3 [0/60000] Loss: 0.173543
CPU Train Epoch: 3 [6400/60000] Loss: 0.202172
CPU Train Epoch: 3 [12800/60000] Loss: 0.136533
CPU Train Epoch: 3 [19200/60000] Loss: 0.134003
CPU Train Epoch: 3 [25600/60000] Loss: 0.132434
CPU Train Epoch: 3 [32000/60000] Loss: 0.094342
CPU Train Epoch: 3 [38400/60000] Loss: 0.100226
CPU Train Epoch: 3 [44800/60000] Loss: 0.144557
CPU Train Epoch: 3 [51200/60000] Loss: 0.198651
CPU Train Epoch: 3 [57600/60000] Loss: 0.168699
CPU Train Epoch: 4 [0/60000] Loss: 0.127727
CPU Train Epoch: 4 [6400/60000] Loss: 0.093230
CPU Train Epoch: 4 [12800/60000] Loss: 0.136490
CPU Train Epoch: 4 [19200/60000] Loss: 0.135140
CPU Train Epoch: 4 [25600/60000] Loss: 0.055327
CPU Train Epoch: 4 [32000/60000] Loss: 0.053846
CPU Train Epoch: 4 [38400/60000] Loss: 0.078890
CPU Train Epoch: 4 [44800/60000] Loss: 0.075155
CPU Train Epoch: 4 [51200/60000] Loss: 0.123335
CPU Train Epoch: 4 [57600/60000] Loss: 0.092369
CPU Train Epoch: 5 [0/60000] Loss: 0.112182
CPU Train Epoch: 5 [6400/60000] Loss: 0.088302
CPU Train Epoch: 5 [12800/60000] Loss: 0.038077
CPU Train Epoch: 5 [19200/60000] Loss: 0.102723
CPU Train Epoch: 5 [25600/60000] Loss: 0.130394
CPU Train Epoch: 5 [32000/60000] Loss: 0.069813
CPU Train Epoch: 5 [38400/60000] Loss: 0.103911
CPU Train Epoch: 5 [44800/60000] Loss: 0.096936
CPU Train Epoch: 5 [51200/60000] Loss: 0.049998
CPU Train Epoch: 5 [57600/60000] Loss: 0.150325
Training Time on CPU: 53.06 seconds
Training on GPU...
GPU Train Epoch: 1 [0/60000] Loss: 0.096387
GPU Train Epoch: 1 [6400/60000] Loss: 0.097615
GPU Train Epoch: 1 [12800/60000] Loss: 0.053897
GPU Train Epoch: 1 [19200/60000] Loss: 0.136676
GPU Train Epoch: 1 [25600/60000] Loss: 0.095318
GPU Train Epoch: 1 [32000/60000] Loss: 0.025371
GPU Train Epoch: 1 [38400/60000] Loss: 0.132306
GPU Train Epoch: 1 [44800/60000] Loss: 0.034585
GPU Train Epoch: 1 [51200/60000] Loss: 0.042037
GPU Train Epoch: 1 [57600/60000] Loss: 0.027025
GPU Train Epoch: 2 [0/60000] Loss: 0.183863
GPU Train Epoch: 2 [6400/60000] Loss: 0.098212
GPU Train Epoch: 2 [12800/60000] Loss: 0.050068
GPU Train Epoch: 2 [19200/60000] Loss: 0.054863
GPU Train Epoch: 2 [25600/60000] Loss: 0.095840
GPU Train Epoch: 2 [32000/60000] Loss: 0.337425
GPU Train Epoch: 2 [38400/60000] Loss: 0.041914
GPU Train Epoch: 2 [44800/60000] Loss: 0.031184
GPU Train Epoch: 2 [51200/60000] Loss: 0.150784
GPU Train Epoch: 2 [57600/60000] Loss: 0.102666
GPU Train Epoch: 3 [0/60000] Loss: 0.287594
GPU Train Epoch: 3 [6400/60000] Loss: 0.044644
GPU Train Epoch: 3 [12800/60000] Loss: 0.109826
GPU Train Epoch: 3 [19200/60000] Loss: 0.121095
GPU Train Epoch: 3 [25600/60000] Loss: 0.048006
GPU Train Epoch: 3 [32000/60000] Loss: 0.190706
GPU Train Epoch: 3 [38400/60000] Loss: 0.049745
GPU Train Epoch: 3 [44800/60000] Loss: 0.058861
GPU Train Epoch: 3 [51200/60000] Loss: 0.010854
GPU Train Epoch: 3 [57600/60000] Loss: 0.091919
GPU Train Epoch: 4 [0/60000] Loss: 0.149170
GPU Train Epoch: 4 [6400/60000] Loss: 0.045755
GPU Train Epoch: 4 [12800/60000] Loss: 0.068857
GPU Train Epoch: 4 [19200/60000] Loss: 0.013966
GPU Train Epoch: 4 [25600/60000] Loss: 0.111864
GPU Train Epoch: 4 [32000/60000] Loss: 0.129941
GPU Train Epoch: 4 [38400/60000] Loss: 0.108321
GPU Train Epoch: 4 [44800/60000] Loss: 0.138987
GPU Train Epoch: 4 [51200/60000] Loss: 0.118738
GPU Train Epoch: 4 [57600/60000] Loss: 0.018990
GPU Train Epoch: 5 [0/60000] Loss: 0.033558
GPU Train Epoch: 5 [6400/60000] Loss: 0.064750
GPU Train Epoch: 5 [12800/60000] Loss: 0.106976
GPU Train Epoch: 5 [19200/60000] Loss: 0.128147
GPU Train Epoch: 5 [25600/60000] Loss: 0.049302
GPU Train Epoch: 5 [32000/60000] Loss: 0.034111
GPU Train Epoch: 5 [38400/60000] Loss: 0.134163
GPU Train Epoch: 5 [44800/60000] Loss: 0.012115
GPU Train Epoch: 5 [51200/60000] Loss: 0.200930
GPU Train Epoch: 5 [57600/60000] Loss: 0.055040
Training Time on GPU: 8.50 seconds
'''