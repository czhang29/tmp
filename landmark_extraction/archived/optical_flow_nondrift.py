import os
import json
import cv2
import numpy as np

# Define paths
mediapipe_dir = "/ADL/annotated_Mediapipe/"
recordings_dir = "/ADL/recordings/"
output_dir = "/ADL/annotated_opticalflow_nodrift/"

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

        # Extract first valid wrist position
        initial_wrist = None
        first_valid_frame = None

        for audio_key in data:
            for video_key in data[audio_key]:
                for frame_idx, landmarks in data[audio_key][video_key].items():
                    if landmarks:
                        if "0" in landmarks:  # Wrist landmark
                            initial_wrist = np.array(landmarks["0"], dtype=np.float32)
                            first_valid_frame = int(frame_idx)
                            break
                if initial_wrist is not None:
                    break
            if initial_wrist is not None:
                break

        if initial_wrist is None:
            continue  # Skip if no valid wrist found

        # Load video
        video_path = os.path.join(video_task_path, json_file.replace(".json", ".mp4"))
        cap = cv2.VideoCapture(video_path)

        output_session_path = os.path.join(output_task_path, session_name)
        os.makedirs(output_session_path, exist_ok=True)

        # Skip frames until first_valid_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_valid_frame)

        # Prepare for Optical Flow (Lucas-Kanade)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_pts = initial_wrist.reshape(-1, 1, 2)

        output_positions = {}

        frame_count = first_valid_frame
        while ret:
            ret, next_frame = cap.read()
            if not ret:
                break

            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, **lk_params)

            if status[0][0] == 1:  # If tracking was successful
                wrist_position = next_pts[0][0].tolist()
                output_positions[frame_count] = wrist_position
                prev_pts = next_pts

                # Draw wrist position on frame
                cv2.circle(next_frame, (int(wrist_position[0]), int(wrist_position[1])), 5, (0, 255, 0), -1)

                # Save frame
                frame_output_path = os.path.join(output_session_path, f"frame_{frame_count}.png")
                cv2.imwrite(frame_output_path, next_frame)

            prev_gray = next_gray
            frame_count += 1

        cap.release()

        # Save tracked positions
        output_json_path = os.path.join(output_task_path, f"{session_name}.json")
        with open(output_json_path, "w") as f:
            json.dump(output_positions, f, indent=4)