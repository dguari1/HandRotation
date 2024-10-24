import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt

# Initialize MediaPipe and YOLO models
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./weights/hand_landmarker.task'),
    num_hands=2,
    running_mode=VisionRunningMode.VIDEO)

model = YOLO('./weights/yolov8n.pt')
modelPose = YOLO("./weights/yolov8n.pt")

def handOrientation(handLandmarks, w, h, side):
    # Calculate hand orientation based on landmarks
    thumb = handLandmarks[4]
    wrist = handLandmarks[0]
    middleFingerTip = handLandmarks[12]

    thumbPos = np.array([thumb.x * w, thumb.y * h])
    wristPos = np.array([wrist.x * w, wrist.y * h])
    middleFingerPos = np.array([middleFingerTip.x * w, middleFingerTip.y * h])

    bottomMid = wristPos
    topMid = middleFingerPos

    x_diff = bottomMid[0] - topMid[0]
    y_diff = bottomMid[1] - topMid[1]

    if side == 'Right':
        orientation = np.arctan2(y_diff, x_diff) * 180 / np.pi
    else:
        orientation = np.arctan2(x_diff, y_diff) * 180 / np.pi

    return orientation

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, data)
    return y

def localizePerson(video):
    # Localize person in the video using YOLO
    video_capture = cv2.VideoCapture(video)
    ret, img = video_capture.read()
    if not ret:
        raise Exception('Failed to read video')
    results = modelPose.predict(cv2.resize(img, (0,0), fx=0.25, fy=0.25), verbose=False)
    video_capture.release()
        
    bInt = []
    center = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if (box.cls.item() == 0) and (box.conf.item() > 0.8):
                b = box.xyxy[0]*4
                temp = np.round(b.cpu().numpy(), 0).astype(int)
                if ((temp[2]-temp[0])/(temp[3]-temp[1])) > 0.3:
                    bInt.append(temp)
                    center.append(np.array([(temp[0] + temp[2])/2, (temp[1] + temp[3])/2]))
    
    if len(bInt) == 0:
        h, w, _ = img.shape
        bint = [0, 0, h, w]
        w = w
        h = h
        return bint, w, h

    secIndex = np.argmin(np.linalg.norm(np.array(center) - np.array([img.shape[1]//2, img.shape[0]//2]), axis=1))
    bint = bInt[secIndex]
    w = bint[3]-bint[1]
    h = bint[2]-bint[0]

    return bint, w, h

def process_video(video_path, hand_side):
    # Process video and predict hand orientations
    detector = HandLandmarker.create_from_options(options)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("The video FPS is zero, cannot process video.")
    bint, w, h = localizePerson(video_path)
    idx = 0
    estOrient = []

    while True:
        success, frame = video.read()
        if not success:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img.shape
        x1 = max(bint[1]-int(w*0.1), 0)
        x2 = min(bint[3]+int(w*0.1), h_img)
        y1 = max(bint[0]-int(h*0.1), 0)
        y2 = min(bint[2]+int(h*0.1), w_img)
        image = img[x1:x2, y1:y2, :]
        imageMediaPipe = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        
        # Calculate timestamp in microseconds as an integer
        timestamp = int(idx * 1e6 / fps)
        detection_result = detector.detect_for_video(imageMediaPipe, timestamp)

        orientation = None
        if len(detection_result.handedness) >= 1:
            for i in range(len(detection_result.handedness)):
                hand_label = detection_result.handedness[i][0].category_name.lower()
                if hand_label == hand_side.lower():
                    orientation = handOrientation(
                        detection_result.hand_landmarks[i],
                        image.shape[1],
                        image.shape[0],
                        hand_label.capitalize())
                    break

        if orientation is None:
            orientation = estOrient[-1] if idx > 0 else 0

        estOrient.append(orientation)
        idx += 1

    video.release()
    estOrient = np.array(estOrient)

    if len(estOrient) > 3:
        smoothEstOrient = bandpass_filter(estOrient, 0.5, 10, fps, order=5)
        smoothEstOrient = (smoothEstOrient + 180) % 360
    else:
        smoothEstOrient = estOrient

    return smoothEstOrient

def save_predictions(output_path, video_name, predictions, hand):
    np.save(os.path.join(output_path, f"{video_name}_{hand}_predictions.npy"), np.array(predictions))

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

def main():
    parser = argparse.ArgumentParser(description='Hand Rotation Angle Prediction from Video')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output files (npy and plots)')
    parser.add_argument('--hand', type=str, choices=['left', 'right'], required=True, help='Hand to predict (left or right)')
    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path
    hand_side = args.hand

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Extract video name without extension for naming outputs
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    predictions = process_video(video_path, hand_side)

    # Save predictions using the specified function and naming convention
    save_predictions(output_path, video_name, predictions, hand_side)

    # Generate frame numbers for plotting
    frame_numbers = np.arange(len(predictions))

    # Plot predictions using the specified function and naming convention
    plot_predictions(output_path, video_name, frame_numbers, predictions, hand_side)

if __name__ == "__main__":
    main()
