import socket
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Define your CNN1D model class here (with a similar architecture to CNN_MODEL.py)
class CNN1D(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=70, out_channels=64, kernel_size=3)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.fc1 = torch.nn.Linear(64 * 9, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 9)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
num_classes = 2  # Set the number of classes (adjust as needed)
model = CNN1D(num_classes)

# Load the trained model weights from the saved file ('your_model.pth')
model.load_state_dict(torch.load('your_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Initialize the label encoder with the same encoder used during training
label_encoder = LabelEncoder()
class_labels = ['Blink', 'Crunch']  # Adjust this list as needed
label_encoder.fit(class_labels)

# Define function to process and predict real-time data
def predict_real_time_data(received_data):
    # Parse the received data and convert it to a NumPy array
    frame_data = np.array(received_data.split(',')).astype(float)
    
    # Assuming frame_data contains 70 values (adjust as needed)
    
    # Process the frame_data, e.g., normalize or preprocess
    
    # Append the frame_data to a buffer
    if 'data_buffer' not in predict_real_time_data.__dict__:
        predict_real_time_data.data_buffer = []

    predict_real_time_data.data_buffer.append(frame_data)

    # Check if the buffer contains 20 instances
    if len(predict_real_time_data.data_buffer) >= 20:
        # Combine the 20 instances into one input tensor
        combined_data = np.stack(predict_real_time_data.data_buffer, axis=2)
        frame_data = torch.FloatTensor(combined_data).view(1, 70, 20)  # Adjust dimensions as needed

        # Make predictions using the loaded model
        with torch.no_grad():
            outputs = model(frame_data.permute(0, 2, 1))
            _, predicted = torch.max(outputs.data, 1)

        # Map the predicted class index back to the label
        predicted_label = label_encoder.inverse_transform(predicted.numpy())[0]
        
        # Clear the data buffer
        predict_real_time_data.data_buffer.clear()

        return predicted_label

    return None

# Set up UDP socket for real-time data reception
UDP_IP = "127.0.0.1"
UDP_PORT = 1000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    received_data = data.decode().strip()
    
    # Perform real-time prediction
    predicted_label = predict_real_time_data(received_data)
    
    if predicted_label:
        print("Received data:", received_data)
        print("Predicted label:", predicted_label)
