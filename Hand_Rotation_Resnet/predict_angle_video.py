import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# Define ResNet Regression Model
class ResNetForRegression(nn.Module):
    def __init__(self):
        super(ResNetForRegression, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Using ResNet-50
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Output a single continuous value for regression

    def forward(self, x):
        return self.resnet(x)

# Load YOLO model
def load_yolo_model():
    model = YOLO('./weights/best.pt')  # Path to the best YOLO weights
    return model

# Load ResNet regression model
def load_regression_model():
    model = ResNetForRegression()
    model.load_state_dict(torch.load('./weights/resnet_best_val_loss.pth', map_location='cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Process a single video and return predictions
def process_single_video(video_path, hand, model, regression_model, device):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], [], []

    frame_numbers = []
    predictions = []

    frame_id = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        # Filter detections based on the specified hand
        if hand == 'left':
            hand_detections = [d for d in results[0].boxes if int(d.cls) == 0]
        else:
            hand_detections = [d for d in results[0].boxes if int(d.cls) == 1]

        # Get the highest confidence detection
        best_hand = max(hand_detections, key=lambda x: x.conf) if hand_detections else None

        # Predict angle using ResNet
        if best_hand:
            hand_box = best_hand.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, hand_box[:4])
            hand_crop = frame[y1:y2, x1:x2]
            angle = predict_angle(hand_crop, regression_model, transform, device)
            predictions.append(angle)
        else:
            predictions.append(None)

        frame_numbers.append(frame_id)
        frame_id += 1

    video.release()
    return frame_numbers, predictions

# Save predictions to .npy files
def save_predictions(output_path, video_name, predictions, hand):
    np.save(os.path.join(output_path, f"{video_name}_{hand}_predictions.npy"), np.array(predictions))

# Plot and save graphs for ResNet predictions
def plot_predictions(output_path, video_name, frame_numbers, predictions, hand):
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, predictions, label=f'{hand.capitalize()} Hand Predictions', color="blue")
    plt.xlabel('Frame Number')
    plt.ylabel('Predicted Angle')
    plt.title(f'{hand.capitalize()} Hand Angle Predictions for {video_name}')
    plt.ylim(0, 360)
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{video_name}_{hand}_predictions.png"))
    plt.close()

# Main function to process the video
def process_video(video_path, output_path, hand):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_path, exist_ok=True)

    # Load models
    yolo_model = load_yolo_model()
    regression_model, device = load_regression_model()

    # Get predictions from the video
    frame_numbers, predictions = process_single_video(video_path, hand, yolo_model, regression_model, device)

    # Save predictions and plots
    save_predictions(output_path, video_name, predictions, hand)
    plot_predictions(output_path, video_name, frame_numbers, predictions, hand)

    print(f"Processed {hand} hand for video: {video_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hand Rotation Angle Prediction from Video')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output files (npy and plots)')
    parser.add_argument('--hand', type=str, choices=['left', 'right'], required=True, help='Hand to predict (left or right)')

    args = parser.parse_args()
    process_video(args.video_path, args.output_path, args.hand)
