import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import math
import matplotlib.pyplot as plt

# Load data from JSON file
output_file_path = '[PROCESSED_RESULTS_JSON_PATH]'
with open(output_file_path, "r") as infile:
    jdict = json.load(infile)

# Function to convert angle to sine and cosine components
def angle_to_sin_cos(angle):
    angle_rad = math.radians(angle)
    return math.sin(angle_rad), math.cos(angle_rad)

# Define Dataset class
class RawAngleDataset(Dataset):
    def __init__(self, jdict, split='train', padding_value=0.0, pad_size=16):
        self.data = []
        self.padding_value = angle_to_sin_cos(padding_value)  # Normalize padding value
        self.left_pad = pad_size // 2 - 1
        self.right_pad = pad_size // 2

        for video_name, video_data in jdict[split].items():
            left_pred = video_data['left']['pred']
            left_anno = video_data['left']['anno']
            right_pred = video_data['right']['pred']
            right_anno = video_data['right']['anno']

            # Process both left and right data
            self.process_angles(left_pred, left_anno)
            self.process_angles(right_pred, right_anno)

    def process_angles(self, preds, annos):
        num_angles = len(preds)
        for i in range(num_angles):
            # Get past frames with left padding
            if i < self.left_pad:
                past_frames = [self.padding_value] * (self.left_pad - i) + \
                              [angle_to_sin_cos(p) for p in preds[:i]]
            else:
                past_frames = [angle_to_sin_cos(p) for p in preds[i - self.left_pad:i]]

            # Get future frames with right padding
            if i + self.right_pad >= num_angles:
                future_frames = [angle_to_sin_cos(p) for p in preds[i + 1:]] + \
                                [self.padding_value] * (i + self.right_pad + 1 - num_angles)
            else:
                future_frames = [angle_to_sin_cos(p) for p in preds[i + 1:i + self.right_pad + 1]]

            # Concatenate past, current, and future frames
            current_frame = angle_to_sin_cos(preds[i])
            input_window = past_frames + [current_frame] + future_frames  # List of tuples (sin, cos)
            sin_components, cos_components = zip(*input_window)  # Separate sin and cos components

            # Target angle
            target_sin, target_cos = angle_to_sin_cos(annos[i])
            self.data.append((
                np.array([sin_components, cos_components], dtype=np.float32).T,  # Shape: (16, 2)
                np.array([target_sin, target_cos], dtype=np.float32)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_window, target = self.data[idx]
        return torch.tensor(input_window), torch.tensor(target)

# Define Angular Distance Loss function
def angular_distance_loss(output, target):
    output_norm = output / output.norm(dim=1, keepdim=True)
    target_norm = target / target.norm(dim=1, keepdim=True)
    cos_theta = (output_norm * target_norm).sum(dim=1)
    loss = 1 - cos_theta.mean()
    return loss

# Define Deep BiLSTM Model
class DeepBiLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=4,
                 output_size=2, dropout=0.4):
        super(DeepBiLSTMModel, self).__init__()

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Fully connected layers with BatchNorm
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = x.view(-1, 16, 2)  # Input shape: (batch_size, 16, 2)

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        lstm_out_current = lstm_out[:, 7, :]  # Extract the 8th frame output

        # Fully connected layers
        x = self.fc1(lstm_out_current)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        output = self.fc3(x)

        return output  # Output shape: (batch_size, 2)

# Define evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_angle_error = 0.0
    threshold = 5.0  # Adjusted threshold for angle difference in degrees

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            outputs = model(imgs)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert sine and cosine to angles
            preds_angle = torch.atan2(outputs[:, 0], outputs[:, 1]) * (180 / math.pi)
            labels_angle = torch.atan2(labels[:, 0], labels[:, 1]) * (180 / math.pi)

            # Adjust angles to [0, 360)
            preds_angle = preds_angle % 360
            labels_angle = labels_angle % 360

            # Calculate angle difference
            angle_diff = torch.abs(preds_angle - labels_angle)
            angle_diff = torch.min(angle_diff, 360 - angle_diff)  # Cyclical adjustment

            # Update metrics
            correct += (angle_diff < threshold).sum().item()
            total += labels.size(0)
            total_angle_error += angle_diff.sum().item()

    avg_loss = total_loss / len(val_loader)
    val_acc = correct / total
    avg_angle_error = total_angle_error / total
    return val_acc, avg_loss, avg_angle_error

# Define training function
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=50, patience=10, model_save_dir='[MODEL_SAVE_DIRECTORY]'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_dir, 'best_val_loss.pth')
    epochs_no_improve = 0

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss /= len(train_loader)

        # Evaluate the model
        val_acc, avg_val_loss, avg_angle_error = evaluate(model, val_loader, criterion, device)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.5f}, "
              f"Val Acc: {val_acc:.5f}, Val Loss: {avg_val_loss:.4f}, "
              f"Avg Angle Error: {avg_angle_error:.2f} degrees")

        # Save model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved with Best Val Loss: {best_val_loss:.4f}")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}. '
                  f'No improvement for {patience} consecutive epochs.')
            break

    return best_model_path

# Prepare data loaders
train_dataset = RawAngleDataset(jdict, split='train', padding_value=0.0, pad_size=16)
val_dataset = RawAngleDataset(jdict, split='val', padding_value=0.0, pad_size=16)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Instantiate the model
model = DeepBiLSTMModel(input_size=2, hidden_size=128, num_layers=4, output_size=2, dropout=0.4)

# Define loss and optimizer
criterion = angular_distance_loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Train the model and get the best model path
best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, model_save_dir='[MODEL_SAVE_DIRECTORY]')

# Load the best model based on validation loss
model.load_state_dict(torch.load(best_model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Loaded best model from {best_model_path}")

# Generate a single combined plot
for k, v in jdict['val'].items():
    for k2 in ['left', 'right']:
        pred_seq = torch.tensor(v[k2]['pred'], dtype=torch.float32)
        anno_seq = torch.tensor(v[k2]['anno'], dtype=torch.float32)

        # Initial predictions before LSTM (raw predictions)
        initial_preds = pred_seq.numpy() % 360  # Ensure within [0, 360)

        # Predictions after LSTM
        lstm_preds = predict(model, pred_seq)

        # True annotations
        true_angles = anno_seq.numpy() % 360  # Ensure within [0, 360)

        # Single combined plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_angles, label='True Angles', linewidth=2)
        plt.plot(initial_preds, label='Initial Predictions', linestyle='--')
        plt.plot(lstm_preds, label='LSTM Predictions', linestyle=':')
        plt.title(f'Comparison of True Angles, Initial Predictions, and LSTM Predictions for {k} {k2}')
        plt.xlabel('Frame Index')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.show()


