import os
import json
import av
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


def process_videos(input_dir, output_dir):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        annotated_subdir = os.path.join(output_dir, subdir)
        os.makedirs(annotated_subdir, exist_ok=True)

        for video_file in os.listdir(subdir_path):
            if not video_file.endswith('.mp4'):
                continue

            video_path = os.path.join(subdir_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(annotated_subdir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)

            json_data = {subdir: {video_file: {}}}

            with av.open(video_path, 'r') as container:
                stream = container.streams.video[0]
                frame_idx = 1

                try:
                    for frame in tqdm(container.decode(stream), desc=f'Processing {video_file}'):
                        frame_np = frame.to_ndarray(format='rgb24')
                        annotated_frame = frame_np.copy()

                        image_rgb = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                        pose_results = pose.process(image_rgb)

                        frame_data = {}

                        if pose_results.pose_landmarks:
                            h, w, _ = frame_np.shape
                            hand_landmarks = {idx: (int(pose_results.pose_landmarks.landmark[idx].x * w),
                                                    int(pose_results.pose_landmarks.landmark[idx].y * h))
                                              for idx in [15, 16]}  # Left Hand (15) & Right Hand (16)

                            elbows = {idx: (int(pose_results.pose_landmarks.landmark[idx].x * w),
                                            int(pose_results.pose_landmarks.landmark[idx].y * h))
                                      for idx in [13, 14]}  # Left Elbow (13) & Right Elbow (14)

                            # Determine which hands are detected
                            hand_status = 0  # 0: None, 1: Left, 2: Right, 3: Both
                            if 15 in hand_landmarks and 16 in hand_landmarks:
                                hand_status = 3
                            elif 15 in hand_landmarks:
                                hand_status = 1
                            elif 16 in hand_landmarks:
                                hand_status = 2

                            frame_data['hand_status'] = hand_status
                            frame_data['hand_landmarks'] = hand_landmarks
                            frame_data['elbows'] = elbows

                            # Draw landmarks
                            for _, (x, y) in hand_landmarks.items():
                                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                            for _, (x, y) in elbows.items():
                                cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)

                        json_data[subdir][video_file][str(frame_idx)] = frame_data

                        frame_output_path = os.path.join(video_output_dir, f'frame_{frame_idx:04d}.png')
                        cv2.imwrite(frame_output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                        frame_idx += 1
                except av.error.EOFError as e:
                    print(f'PyAV audio stream EOF error ({e})')

            json_output_path = os.path.join(annotated_subdir, f'{video_name}.json')
            with open(json_output_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    input_directory = "/home/czhang/PycharmProjects/ModalityAI/ADL/recordings"
    output_directory = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_hand_elbow_data"
    process_videos(input_directory, output_directory)
