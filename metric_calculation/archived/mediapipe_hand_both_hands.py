import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean


def calculate_metrics(positions):
    velocities = [euclidean(positions[i], positions[i - 1]) for i in range(1, len(positions)) if
                  positions[i - 1] and positions[i]]

    if len(velocities) < 2:
        return None, None, None

    accelerations = np.gradient(velocities) if velocities else []
    jerks = np.gradient(accelerations) if len(accelerations) > 1 else []

    return (
        np.mean(velocities) if velocities else 0,
        np.mean(accelerations) if len(accelerations) > 1 else 0,
        np.mean(jerks) if len(jerks) > 1 else 0,
    )


def process_json_files(base_dir, drop_mode, output_file, task_filter, handedness_csv=None):
    results = []

    handedness_df = None
    if handedness_csv and drop_mode != 'None':
        try:
            handedness_df = pd.read_csv(handedness_csv)
            print(handedness_df.head())  # Display the first few rows
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return  # Stop the function if reading fails

        handedness_df['Bad_Session_Drop'] = handedness_df['Bad_Session_Drop'].fillna(0)
        handedness_df['No_landmark_Drop'] = handedness_df['No_landmark_Drop'].fillna(0)

    for subdir, _, files in os.walk(base_dir):
        task_type = (
            'Face' if 'face' in subdir.lower() else
            'Teeth' if 'teeth' in subdir.lower() else
            'Hair' if 'hair' in subdir.lower() else None
        )

        # Skip processing files that do not match the current task filter
        if task_type != task_filter:
            continue

        for file in files:
            if not file.endswith('.json'):
                continue

            session_id = file.split('_')[2]
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)

            positions = {15: [], 16: []}
            for frames in data.values():
                for video_frames in frames.values():
                    for frame_data in video_frames.values():
                        for hand in [15, 16]:
                            if str(hand) in frame_data.get('hand_landmarks', {}):
                                positions[hand].append(tuple(frame_data['hand_landmarks'][str(hand)]))

            print(f"Positions for session {session_id}: {positions}")

            # Calculate metrics for left hand (15) and right hand (16)
            left_hand_metrics = calculate_metrics(positions[15])
            right_hand_metrics = calculate_metrics(positions[16])

            # Take the average of metrics from both hands
            velocity = np.mean(
                [m for m in [left_hand_metrics[0], right_hand_metrics[0]] if m is not None]
            ) if any(m is not None for m in [left_hand_metrics[0], right_hand_metrics[0]]) else None

            acceleration = np.mean(
                [m for m in [left_hand_metrics[1], right_hand_metrics[1]] if m is not None]
            ) if any(m is not None for m in [left_hand_metrics[1], right_hand_metrics[1]]) else None

            jerk = np.mean(
                [m for m in [left_hand_metrics[2], right_hand_metrics[2]] if m is not None]
            ) if any(m is not None for m in [left_hand_metrics[2], right_hand_metrics[2]]) else None

            # Append results for the session
            results.append({
                'Session_ID': session_id,
                'Task_Type': task_type,
                'Left_Hand_Velocity': left_hand_metrics[0],
                'Right_Hand_Velocity': right_hand_metrics[0],
                'Velocity': velocity,
                'Left_Hand_Acceleration': left_hand_metrics[1],
                'Right_Hand_Acceleration': right_hand_metrics[1],
                'Acceleration': acceleration,
                'Left_Hand_Jerk': left_hand_metrics[2],
                'Right_Hand_Jerk': right_hand_metrics[2],
                'Jerk': jerk
            })

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_file, index=False)


def main():
    base_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_entire_hand_nogbr"
    handedness_csv = os.path.join(base_dir, 'handedness_annotations_20250313.csv')
    print(handedness_csv)

    drop_mode = input("Enter drop mode ('No_landmark_Drop', 'Bad_Session_Drop', or 'None'): ")

    # Process Face task and save only Face sessions to metrics_face.csv
    process_json_files(
        base_dir,
        drop_mode,
        os.path.join(base_dir, '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/no_landmark_drop/metrics_face.csv'),
        task_filter='Face',
        handedness_csv=handedness_csv
    )

    # Process Teeth task and save only Teeth sessions to metrics_teeth.csv
    process_json_files(
        base_dir,
        drop_mode,
        os.path.join(base_dir, '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/no_landmark_drop/metrics_teeth.csv'),
        task_filter='Teeth',
        handedness_csv=handedness_csv
    )

    # Process Hair task and save only Hair sessions to metrics_hair.csv
    process_json_files(
        base_dir,
        drop_mode,
        os.path.join(base_dir, '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/no_landmark_drop/metrics_hair.csv'),
        task_filter='Hair',
        handedness_csv=handedness_csv
    )


if __name__ == "__main__":
    main()
