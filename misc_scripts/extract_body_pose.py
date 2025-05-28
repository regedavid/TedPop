import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# === CONFIGURATION ===
video_dir = "/Users/david.urdea/Desktop/personal/TedPop/downloads"
output_dir = "/Users/david.urdea/Desktop/personal/TedPop/pose_features"
os.makedirs(output_dir, exist_ok=True)
frame_skip = 30         # Process every 30th frame (~1 fps)
max_duration = 120      # Limit to 2 minutes of video
num_workers = int(cpu_count() * 0.7)

# === INIT POSE DETECTOR (STATIC FOR WORKERS) ===
mp_pose = mp.solutions.pose

def extract_features(landmarks):
    features = []
    if not landmarks:
        return [0.0] * 20

    keypoints = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]

    for kp in keypoints:
        lm = landmarks[kp.value]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return features

def process_video(filename):
    if not filename.endswith(".mp4"):
        return

    video_path = os.path.join(video_dir, filename)
    out_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npy")

    if os.path.exists(out_path):
        return  # Already processed

    print(f"Processing video: {filename}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(min(max_duration * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    all_features = []
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    frame_id = 0
    while cap.isOpened() and frame_id < max_frames:
        success, frame = cap.read()
        if not success:
            break

        if frame_id % frame_skip == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks.landmark)
                all_features.append(features)

        frame_id += 1

    cap.release()

    mean_features = np.mean(all_features, axis=0) if all_features else np.zeros(20)
    np.save(out_path, mean_features)

def main():
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

    print(f"Processing {len(video_files)} videos using {num_workers} workers...")
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, video_files), total=len(video_files)))

    print("Pose feature extraction completed.")

if __name__ == "__main__":
    main()