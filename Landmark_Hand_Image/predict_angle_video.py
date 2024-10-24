import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO
from scipy.signal import butter, filtfilt
import mediapipe as mp

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Initialize YOLO models
# Adjust the paths to your models as necessary
yolo_hand_model = YOLO('./weights/best.pt')  # Path to your trained hand detection model
yolo_pose_model = YOLO('./weights/yolov8n.pt')  # Path to your person detection model

def handOrientation(hand_landmarks, img_width, img_height, hand_side):
    """
    Calculate the orientation angle of a hand based on the wrist and thumb landmarks.

    Parameters:
    - hand_landmarks: Detected hand landmarks from MediaPipe.
    - img_width: Width of the hand image.
    - img_height: Height of the hand image.
    - hand_side: 'Left' or 'Right' indicating the hand side.

    Returns:
    - angle: Angle in degrees within [0, 360].
    """
    # Get thumb (landmark 4) and wrist (landmark 0) for calculating orientation
    thumb = hand_landmarks.landmark[4]
    wrist = hand_landmarks.landmark[0]

    thumb_pos = np.array([thumb.x * img_width, thumb.y * img_height])
    wrist_pos = np.array([wrist.x * img_width, wrist.y * img_height])

    # Calculate the angle between the wrist and the thumb
    angle = np.arctan2(thumb_pos[1] - wrist_pos[1], thumb_pos[0] - wrist_pos[0]) * (180 / np.pi)
    angle = (angle + 360) % 360  # Normalize to [0, 360]

    return angle

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    Apply a Butterworth bandpass filter to the data.

    Parameters:
    - data: The input signal data.
    - lowcut: Low cutoff frequency.
    - highcut: High cutoff frequency.
    - fs: Sampling frequency (e.g., frames per second).
    - order: Order of the filter.

    Returns:
    - y: The filtered data.
    """
    nyquist = 0.5 * fs
    if nyquist <= 0:
        return data  # Avoid division by zero
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1:
        high = 0.99  # Ensure highcut is less than Nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def localize_person_frame(frame):
    """
    Localize the person in a frame using the YOLO model.

    Parameters:
    - frame: The input image frame.

    Returns:
    - bint: Bounding box coordinates of the person [x_min, y_min, x_max, y_max].
    - w: Width of the bounding box.
    - h: Height of the bounding box.
    """
    img_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    results = yolo_pose_model(img_small, verbose=False)
    b_int = []
    centers = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if (box.cls.item() == 0) and (box.conf.item() > 0.8):  # Class 0 is person
                b = box.xyxy[0] * 4  # Scale back to original size
                temp = np.round(b.cpu().numpy(), 0).astype(int)
                # Ensure the bounding box is valid
                if temp[2] > temp[0] and temp[3] > temp[1]:
                    aspect_ratio = (temp[2] - temp[0]) / (temp[3] - temp[1])
                    if aspect_ratio > 0.3:
                        b_int.append(temp)
                        centers.append(np.array([
                            (temp[0] + temp[2]) / 2,
                            (temp[1] + temp[3]) / 2
                        ]))

    if len(b_int) == 0:
        h, w, _ = frame.shape
        bint = [0, 0, w, h]
        return bint, w, h

    # Select the person closest to the center
    frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    distances = [np.linalg.norm(center - frame_center) for center in centers]
    sec_index = np.argmin(distances)
    bint = b_int[sec_index]
    w = bint[2] - bint[0]
    h = bint[3] - bint[1]

    return bint, w, h

def process_video(video_path, hand_side):
    """
    Process the video to estimate the hand rotation angles.

    Parameters:
    - video_path: Path to the input video file.
    - hand_side: 'Left' or 'Right' indicating which hand to process.

    Returns:
    - smooth_est_orient: Smoothed array of orientation angles.
    """
    hand_side = hand_side.capitalize()  # Ensure 'Left' or 'Right'
    # Verify that the video file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("The video FPS is zero, cannot process video.")

    est_orient = []
    idx = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if idx == 0:
            # Localize person in the first frame
            bint, w, h = localize_person_frame(frame)
            x1 = max(bint[0] - int(w * 0.1), 0)
            x2 = min(bint[2] + int(w * 0.1), frame.shape[1])
            y1 = max(bint[1] - int(h * 0.1), 0)
            y2 = min(bint[3] + int(h * 0.1), frame.shape[0])

        # Crop the frame to the person region
        cropped_frame = frame[y1:y2, x1:x2]

        # Run hand detection using YOLO
        results = yolo_hand_model(cropped_frame, verbose=False)
        hand_detections = []

        for detection in results[0].boxes:
            class_id = int(detection.cls)
            confidence = detection.conf.item()
            # Adjust class IDs based on your model's training
            # Assuming class_id 0 for left hand and 1 for right hand
            if confidence > 0.5:
                if hand_side == 'Left' and class_id == 0:
                    hand_detections.append(detection)
                elif hand_side == 'Right' and class_id == 1:
                    hand_detections.append(detection)

        if hand_detections:
            # Get the best detection
            best_hand = max(hand_detections, key=lambda x: x.conf)
            hand_box = best_hand.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = map(int, hand_box)

            # Ensure coordinates are within image bounds
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, cropped_frame.shape[1])
            y_max = min(y_max, cropped_frame.shape[0])

            # Crop the hand region
            hand_crop = cropped_frame[y_min:y_max, x_min:x_max]

            # Convert to RGB and process with MediaPipe
            hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            hand_results = hands_detector.process(hand_crop_rgb)

            if hand_results.multi_hand_landmarks:
                # Assuming the first detected hand is the one we want
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                orientation = handOrientation(
                    hand_landmarks,
                    hand_crop.shape[1],
                    hand_crop.shape[0],
                    hand_side
                )
            else:
                print(f"No hand landmarks detected in frame {idx}.")
                orientation = est_orient[-1] if est_orient else 0
        else:
            print(f"No hand detections in frame {idx}.")
            orientation = est_orient[-1] if est_orient else 0

        est_orient.append(orientation)
        idx += 1

    video.release()
    hands_detector.close()  # Release MediaPipe resources
    est_orient = np.array(est_orient)

    # Apply angle unwrapping to eliminate sudden flips
    est_orient_rad = np.radians(est_orient)
    unwrapped_orient_rad = np.unwrap(est_orient_rad)
    unwrapped_orient_deg = np.degrees(unwrapped_orient_rad)

    # Apply bandpass filter to smooth the estimated orientations
    if len(unwrapped_orient_deg) > 3:
        # Adjust filter parameters as needed
        smooth_unwrapped_orient_deg = bandpass_filter(unwrapped_orient_deg, 0.1, 3.0, fps, order=3)
    else:
        smooth_unwrapped_orient_deg = unwrapped_orient_deg

    # Normalize angles to [0, 360] degrees
    smooth_est_orient = smooth_unwrapped_orient_deg % 360

    return smooth_est_orient

def save_predictions(output_path, video_name, predictions, hand):
    """
    Save the predicted orientation angles to a NumPy file.

    Parameters:
    - output_path: Directory where the output file will be saved.
    - video_name: Name of the video (without extension).
    - predictions: Array of predicted orientation angles.
    - hand: 'left' or 'right' indicating which hand was processed.
    """
    output_file = os.path.join(output_path, f"{video_name}_{hand}_predictions.npy")
    np.save(output_file, predictions)
    print(f"Predictions saved to {output_file}")

def plot_predictions(output_path, video_name, frame_numbers, predictions, hand):
    """
    Plot the orientation predictions and save the plot as an image.

    Parameters:
    - output_path: Directory where the plot image will be saved.
    - video_name: Name of the video (without extension).
    - frame_numbers: Array of frame numbers corresponding to the predictions.
    - predictions: Array of predicted orientation angles.
    - hand: 'left' or 'right' indicating which hand was processed.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, predictions, label=f'{hand.capitalize()} Hand Predictions', color="blue")
    plt.xlabel('Frame Number')
    plt.ylabel('Predicted Angle (Degrees)')
    plt.title(f'{hand.capitalize()} Hand Angle Predictions for {video_name}')
    plt.ylim(0, 360)
    plt.legend()
    output_file = os.path.join(output_path, f"{video_name}_{hand}_predictions.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Hand Rotation Angle Prediction from Video')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output files (npy and plots)')
    parser.add_argument('--hand', type=str, choices=['left', 'right'], required=True, help='Hand to predict (left or right)')
    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path
    hand_side = args.hand.capitalize()  # Ensure 'Left' or 'Right'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Extract video name without extension for naming outputs
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print("Processing video...")
    predictions = process_video(video_path, hand_side)
    print("Video processing completed.")

    # Save predictions
    save_predictions(output_path, video_name, predictions, hand_side.lower())

    # Generate frame numbers for plotting
    frame_numbers = np.arange(len(predictions))

    # Plot predictions
    plot_predictions(output_path, video_name, frame_numbers, predictions, hand_side.lower())

    print("Processing completed.")

if __name__ == "__main__":
    main()
