import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch.backends
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import math
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Function to convert angle to sine and cosine components
def angle_to_sin_cos(angle):
    angle_rad = math.radians(angle)
    return math.sin(angle_rad), math.cos(angle_rad)

# Define ResNet Regression Model
class ResNetForRegression(nn.Module):
    def __init__(self):
        super(ResNetForRegression, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)

# Define Deep BiLSTM Model
class DeepBiLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=4, output_size=2, dropout=0.4):
        super(DeepBiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn4 = nn.BatchNorm1d(hidden_size // 4)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(hidden_size // 4, output_size)

    def forward(self, x):
        x = x.view(-1, 16, 2)  # Input shape: (batch_size, 16, 2)

        lstm_out, _ = self.lstm(x)
        lstm_out_current = lstm_out[:, 7, :]  # Extract the 8th frame output

        x = self.fc1(lstm_out_current)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        output = self.fc5(x)
        return output  # Output shape: (batch_size, 2)

# Load YOLO model
def load_yolo_model():
    model = YOLO('weights/best.pt')
    #if cuda available, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if mps available use it    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model

# Load ResNet regression model
def load_regression_model():
    model = ResNetForRegression()
    model.load_state_dict(torch.load('weights/resnet_best_val_loss.pth', map_location='cpu'))
    model.eval()
    #if cuda available, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if mps available use it    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model, device

# Load LSTM model
def load_lstm_model():
    model = DeepBiLSTMModel(input_size=2, hidden_size=128, num_layers=4, output_size=2, dropout=0.4)
    model.load_state_dict(torch.load('weights/best_val_loss_lstm.pth', map_location='cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if mps available use it    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model, device

# Image transformation for ResNet input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Predict angle using the ResNet regression model
def predict_angle(crop, model, transform, device):
    crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_tensor = transform(crop_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(crop_tensor).item()
    return prediction

# Apply LSTM model to a sequence of angle predictions
def apply_lstm_to_sequence(predictions, lstm_model, device):
    processed_preds = [0.0 if p is None else p for p in predictions]
    sin_cos_sequence = [angle_to_sin_cos(p) for p in processed_preds]
    sin_cos_sequence = torch.tensor(sin_cos_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    lstm_preds = []
    lstm_model.eval()
    with torch.no_grad():
        for i in range(len(predictions) - 15):
            window = sin_cos_sequence[:, i:i+16, :]
            output = lstm_model(window)
            pred_angle = torch.atan2(output[0, 0], output[0, 1]) * (180 / math.pi)
            pred_angle = pred_angle.item() % 360
            lstm_preds.append(pred_angle)

    lstm_preds = lstm_preds + [None] * (len(predictions) - len(lstm_preds))
    return lstm_preds

# Save both ResNet and LSTM predictions to .npy files
def save_predictions(output_path, video_name, resnet_predictions, lstm_predictions, hand):
    np.save(os.path.join(output_path, f"{video_name}_{hand}_resnet_predictions.npy"), np.array(resnet_predictions))
    np.save(os.path.join(output_path, f"{video_name}_{hand}_lstm_predictions.npy"), np.array(lstm_predictions))

# Plot and save graphs for ResNet and LSTM predictions
def plot_predictions(output_path, video_name, frame_numbers, resnet_preds, lstm_preds, hand):
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, resnet_preds, label="ResNet Predictions", color="blue")
    plt.plot(frame_numbers, lstm_preds, label="LSTM Predictions", color="green")
    plt.xlabel('Frame Number')
    plt.ylabel('Predicted Angle')
    plt.title(f'{hand.capitalize()} Hand Angle Predictions for {video_name}')
    plt.ylim(0, 360)
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{video_name}_{hand}_predictions.png"))
    plt.close()

# Process a single video for a specified hand
def process_single_video(video_path, output_path, hand):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_path, exist_ok=True)

    # Load models
    yolo_model = load_yolo_model()
    regression_model, device = load_regression_model()
    lstm_model, lstm_device = load_lstm_model()

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_id = 0
    frame_numbers = []
    resnet_predictions = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Run YOLO inference
        results = yolo_model(frame, verbose=False)

        # Filter detections based on the specified hand
        if hand == 'left':
            hand_detections = [d for d in results[0].boxes if int(d.cls) == 0]
        else:
            hand_detections = [d for d in results[0].boxes if int(d.cls) == 1]

        # Get the highest confidence detection for the specified hand
        best_hand = max(hand_detections, key=lambda x: x.conf) if hand_detections else None

        # Predict angles using ResNet
        if best_hand:
            hand_box = best_hand.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, hand_box[:4])
            hand_crop = frame[y1:y2, x1:x2]
            angle = predict_angle(hand_crop, regression_model, transform, device)
            resnet_predictions.append(angle)
        else:
            resnet_predictions.append(None)

        frame_numbers.append(frame_id)
        frame_id += 1

    video.release()

    # Apply LSTM model to refine predictions
    lstm_predictions = apply_lstm_to_sequence(resnet_predictions, lstm_model, lstm_device)

    # Save both ResNet and LSTM predictions
    save_predictions(output_path, video_name, resnet_predictions, lstm_predictions, hand)

    # Plot both ResNet and LSTM predictions
    plot_predictions(output_path, video_name, frame_numbers, resnet_predictions, lstm_predictions, hand)

    print(f"Processed {hand} hand for video: {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Rotation Angle Prediction from Video')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output files (npy and plots)')
    parser.add_argument('--hand', type=str, choices=['left', 'right'], required=True, help='Hand to predict (left or right)')

    args = parser.parse_args()
    st = time.time()
    process_single_video(args.video_path, args.output_path, args.hand)
    print(f"Total time taken: {time.time()-st} seconds")