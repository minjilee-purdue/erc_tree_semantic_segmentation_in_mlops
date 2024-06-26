import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)  # y = 2x + 1 + noise

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define a simple feedforward neural network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc = nn.Linear(1, 1)  # Single input feature and single output feature

    def forward(self, x):
        return self.fc(x)

# Instantiate the neural network
model = FeedforwardNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the neural network
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the original data and the fitted line
plt.scatter(X, y, label='Original data')
plt.plot(X, model(X_tensor).detach().numpy(), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Fitted Line by Feedforward Neural Network')
plt.legend()
plt.show()
