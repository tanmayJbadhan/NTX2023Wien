import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your temporal data from a CSV file (replace 'your_data.csv' with your actual data file)
# Load data from CSV files
blink_data = pd.read_csv('eeg_data_blink.csv')
crunch_data = pd.read_csv('eeg_data_crunch.csv')

# Concatenate the data from both files
data = pd.concat([blink_data, crunch_data], ignore_index=True)

# Define the sequence length (number of frames in each sequence)
sequence_length = 20

# Create sequences of 20 frames each
X, y = [], []

for i in range(len(data) - sequence_length + 1):
    X.append(data.iloc[i:i+sequence_length, :-1].values)
    y.append(data.iloc[i+sequence_length-1, -1])

X = np.array(X)
y = np.array(y)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create DataLoader for training and testing data
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=70, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 9, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 9)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
num_classes = len(label_encoder.classes_)
model = CNN1D(num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 2, 1))  # Permute the input to match the expected shape
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {total_loss / len(train_loader)}')

# Evaluate the model on the test data
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.permute(0, 2, 1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on Test Data: {accuracy}%')

# Save the trained model to a file
torch.save(model.state_dict(), 'your_model.pth')
