import os
import json
import cv2
import numpy as np

# Define paths
mediapipe_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_hand_data_included/"
recordings_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/recordings/"
output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_opticalflow_drift_handedness/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over task folders containing 'Face' in the name
for task_name in os.listdir(mediapipe_dir):
    if "Face" not in task_name:
        continue

    task_path = os.path.join(mediapipe_dir, task_name)
    video_task_path = os.path.join(recordings_dir, task_name)
    output_task_path = os.path.join(output_dir, task_name)
    os.makedirs(output_task_path, exist_ok=True)

    for json_file in os.listdir(task_path):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(task_path, json_file)
        session_name = json_file.replace(".json", "")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Load video
        video_path = os.path.join(video_task_path, json_file.replace(".json", ".mp4"))
        cap = cv2.VideoCapture(video_path)

        output_session_path = os.path.join(output_task_path, session_name)
        os.makedirs(output_session_path, exist_ok=True)

        # Prepare for Optical Flow (Lucas-Kanade)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        prev_gray = None
        prev_pts = None
        output_positions = {}
        first_valid_frame = None
        selected_handedness = None

        for audio_key in data:
            for video_key in data[audio_key]:
                for frame_idx, landmarks in sorted(data[audio_key][video_key].items(), key=lambda x: int(x[0])):
                    frame_idx = int(frame_idx)
                    if first_valid_frame is None:
                        for handedness in landmarks:
                            if "0" in landmarks[handedness]:
                                first_valid_frame = frame_idx
                                selected_handedness = handedness
                                break

                    if first_valid_frame is None:
                        continue  # Skip frames until the first valid frame is found

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # If Mediapipe provides a valid landmark, reset tracking
                    if selected_handedness in landmarks and "0" in landmarks[selected_handedness]:
                        prev_pts = np.array(landmarks[selected_handedness]["0"], dtype=np.float32).reshape(-1, 1, 2)

                    # If previous frame exists, perform optical flow
                    if prev_gray is not None and prev_pts is not None:
                        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_pts, None,
                                                                       **lk_params)
                        if status[0][0] == 1:  # If tracking was successful
                            wrist_position = next_pts[0][0].tolist()
                            output_positions[frame_idx] = wrist_position
                            prev_pts = next_pts

                            # Draw wrist position on frame
                            cv2.circle(frame, (int(wrist_position[0]), int(wrist_position[1])), 5, (0, 255, 0), -1)

                    prev_gray = gray_frame

                    # Save frame with the first valid frame index as the starting point
                    frame_output_path = os.path.join(output_session_path, f"frame_{frame_idx - first_valid_frame}.png")
                    cv2.imwrite(frame_output_path, frame)

        cap.release()

        # Save tracked positions
        output_json_path = os.path.join(output_task_path, f"{session_name}.json")
        with open(output_json_path, "w") as f:
            json.dump(output_positions, f, indent=4)
