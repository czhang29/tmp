import os
import json
import av
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


def process_videos(input_dir, output_dir):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

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
                        results = hands.process(image_rgb)
                        frame_data = {}
                        points_to_draw = {}
                        if results.multi_handedness and not results.multi_hand_landmarks:
                            print("Hand detected but no landmarks placed")
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                for idx in [0, 5, 9, 13, 17]:  # Wrist and knuckles
                                    landmark = hand_landmarks.landmark[idx]
                                    h, w, _ = frame_np.shape
                                    x, y = int(landmark.x * w), int(landmark.y * h)
                                    points_to_draw[idx] = (x, y)

                                frame_data[str(frame_idx)] = points_to_draw

                                # Draw points and connect them with a line
                                for i, (point_idx, (x, y)) in enumerate(points_to_draw.items()):
                                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                                    if i > 0:
                                        prev_x, prev_y = list(points_to_draw.values())[i - 1]
                                        cv2.line(annotated_frame, (prev_x, prev_y), (x, y), (255, 0, 0), 2)

                        json_data[subdir][video_file][str(frame_idx)] = points_to_draw

                        frame_output_path = os.path.join(video_output_dir, f'frame_{frame_idx:04d}.png')
                        cv2.imwrite(frame_output_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

                        frame_idx += 1
                except av.error.EOFError as e:
                    print(f'PyAV audio stream EOF error ({e})')

            json_output_path = os.path.join(annotated_subdir, f'{video_name}.json')
            with open(json_output_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    input_directory = "/home/czhang/PycharmProjects/ModalityAI/ADL/recordings"  # Change to your input directory
    output_directory = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand"  # Output directory for annotations
    process_videos(input_directory, output_directory)
