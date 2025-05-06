import os
import json
import av
import numpy as np
import pandas as pd


def convert_timestamps_to_frame_indices(segments, fps):
    """Convert timestamp segments to frame indices."""
    frame_indices = []
    for i in range(0, len(segments), 2):
        start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(segments[i].split(':'))))
        end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(segments[i + 1].split(':'))))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_indices.append((start_frame, end_frame))
    return frame_indices


def calculate_metrics(positions):
    """Calculate velocity, acceleration, and jerk."""
    velocities = [np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
                  for i in range(1, len(positions)) if positions[i - 1] and positions[i]]

    if len(velocities) < 2:
        return None, None, None

    accelerations = np.gradient(velocities) if velocities else []
    jerks = np.gradient(accelerations) if len(accelerations) > 1 else []

    return (
        np.mean(velocities) if velocities else 0,
        np.mean(accelerations) if len(accelerations) > 1 else 0,
        np.mean(jerks) if len(jerks) > 1 else 0,
    )


def process_json_files(base_dir, video_dir, handedness_json_path, output_file):
    # Load handedness annotations
    with open(handedness_json_path, 'r') as f:
        handedness_annotations = json.load(f)

    results = []

    for subdir, _, files in os.walk(base_dir):
        task_type = 'Face' if 'face' in subdir.lower() else None

        if not task_type:
            continue

        for file in files:
            if not file.endswith('.json'):
                continue

            session_id = file.split('_')[2]
            file_path = os.path.join(subdir, file)
            video_name = file.replace('.json', '.mp4')
            video_path = os.path.join(video_dir, os.path.basename(subdir), video_name)

            # Ensure video exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file {video_name} not found in {video_path}")

            # Check Face task annotations
            if video_name not in handedness_annotations:
                raise ValueError(f"Recording {video_name} is missing from handedness_annotation_20250321.json")

            segments = handedness_annotations[video_name]
            if not segments:  # Skip sessions with empty lists
                print(f"Session {video_name} has an empty segment list. Skipping.")
                continue

            # Get FPS of the video
            container = av.open(video_path)
            fps = container.streams.video[0].average_rate
            frame_indices_segments = convert_timestamps_to_frame_indices(segments, fps)

            with open(file_path, 'r') as f:
                data = json.load(f)

            positions_left_hand_segments = []
            positions_right_hand_segments = []

            # Collect landmarks for each segment independently
            for start_frame, end_frame in frame_indices_segments:
                positions_left_hand_segment = []
                positions_right_hand_segment = []

                for frames in data.values():
                    for video_frames in frames.values():
                        for frame_idx_str, frame_data in video_frames.items():
                            frame_idx = int(frame_idx_str)

                            # Skip frames outside the current segment
                            if not (start_frame <= frame_idx <= end_frame):
                                continue

                            # Extract left hand (15) and right hand (16) landmarks
                            if '15' in frame_data.get('hand_landmarks', {}):
                                positions_left_hand_segment.append(tuple(frame_data['hand_landmarks']['15']))
                            if '16' in frame_data.get('hand_landmarks', {}):
                                positions_right_hand_segment.append(tuple(frame_data['hand_landmarks']['16']))

                positions_left_hand_segments.append(positions_left_hand_segment)
                positions_right_hand_segments.append(positions_right_hand_segment)

            # Calculate metrics separately for each segment and average across segments
            left_metrics_per_segment = [
                calculate_metrics(segment) for segment in positions_left_hand_segments
            ]
            right_metrics_per_segment = [
                calculate_metrics(segment) for segment in positions_right_hand_segments
            ]

            # Aggregate metrics across all segments (mean of all segments)
            v_left_mean = np.mean([metrics[0] for metrics in left_metrics_per_segment if metrics[0] is not None])
            a_left_mean = np.mean([metrics[1] for metrics in left_metrics_per_segment if metrics[1] is not None])
            j_left_mean = np.mean([metrics[2] for metrics in left_metrics_per_segment if metrics[2] is not None])

            v_right_mean = np.mean([metrics[0] for metrics in right_metrics_per_segment if metrics[0] is not None])
            a_right_mean = np.mean([metrics[1] for metrics in right_metrics_per_segment if metrics[1] is not None])
            j_right_mean = np.mean([metrics[2] for metrics in right_metrics_per_segment if metrics[2] is not None])

            # Final mean across both hands
            velocity_mean = np.mean([v_left_mean, v_right_mean]) if v_left_mean and v_right_mean else None
            acceleration_mean = np.mean([a_left_mean, a_right_mean]) if a_left_mean and a_right_mean else None
            jerk_mean = np.mean([j_left_mean, j_right_mean]) if j_left_mean and j_right_mean else None

            results.append({
                'Session_ID': session_id,
                'Task_Type': task_type,
                'Velocity': velocity_mean,
                'Acceleration': acceleration_mean,
                'Jerk': jerk_mean,
                'Video_Name': video_name
            })

    pd.DataFrame(results).to_csv(output_file, index=False)


if __name__ == "__main__":
    base_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_entire_hand_nogbr/"
    video_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/recordings"
    handedness_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/hand_detection_annotation_20250321.json"
    output_file = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/hand_detection/metrics_results_face.csv"

    process_json_files(base_dir, video_dir, handedness_json_path, output_file)
