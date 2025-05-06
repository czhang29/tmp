import os
import json
import csv
import numpy as np
from scipy.spatial.distance import euclidean

# Configuration - UPDATE THESE PATHS!
SESSION_HANDEDNESS_PATH = '/home/czhang/PycharmProjects/ModalityAI/ADL/session_handedness.json'  # Verify this path exists!
OUTPUT_CSV = 'adl_face_washing_metrics_pipeline.csv'

# Data directories - keep these as-is if correct
SOURCE_DIRS = [
    '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_no_bgr_confidence_score',
    # '/home/czhang/PycharmProjects/ModalityAI/ADL/crowd_source_mediapipe_hand'
]


def calculate_centroid(landmarks):
    """Calculate centroid of given landmarks."""
    if not landmarks:
        return []
    return np.mean(np.array(landmarks), axis=0).tolist()


def get_wrist_positions(data, hand):
    """Extract wrist positions for specified hand using key landmarks."""
    wrist_positions = {}
    for frame, annotations in data.items():
        if hand in annotations:
            landmarks = [annotations[hand].get(str(i), []) for i in [0, 5, 9, 13, 17]]
            if all(landmarks):
                wrist_positions[int(frame)] = calculate_centroid(landmarks)
    return wrist_positions


def compute_metrics(wrist_positions, total_frames):
    """Compute movement metrics with absolute values."""
    frames = sorted(wrist_positions.keys())
    if len(frames) < 2:
        return [None] * 9

    # Calculate distances using element-wise operations
    distances = []
    prev_pos = wrist_positions[frames[0]]
    for frame in frames[1:]:
        curr_pos = wrist_positions[frame]
        if prev_pos.size and curr_pos.size:  # Check array emptiness properly
            distances.append(euclidean(prev_pos, curr_pos))
        prev_pos = curr_pos

    if not distances:
        return [None] * 9

    # Convert to NumPy arrays for vector operations
    distances = np.array(distances)

    # Use NumPy's gradient and absolute functions
    velocities = np.gradient(distances)
    avg_velocity_abs = np.abs(velocities).mean()

    accelerations = np.gradient(velocities)
    avg_acceleration_abs = np.abs(accelerations).mean() if accelerations.size > 0 else None

    jerks = np.gradient(accelerations) if accelerations.size > 1 else np.array([])
    avg_jerk_abs = np.abs(jerks).mean() if jerks.size > 0 else None

    # Return metrics as native Python floats
    return [
        float(distances.sum()),
        float(velocities.mean()),
        float(avg_velocity_abs),
        float(accelerations.mean()) if accelerations.size > 0 else None,
        float(avg_acceleration_abs) if avg_acceleration_abs is not None else None,
        float(jerks.mean()) if jerks.size > 0 else None,
        float(avg_jerk_abs) if avg_jerk_abs is not None else None,
        len(frames),
        len(frames) / total_frames if total_frames > 0 else 0
    ]


def determine_dominant_hand(session_id, handedness_data):
    """Determine dominant hand from handedness data."""
    counts = {'L': 0, 'R': 0}
    for entry in handedness_data.get(session_id, []):
        for hand in entry:
            counts[hand] += 1
    return max(counts, key=counts.get) if counts['L'] != counts['R'] else 'R'


def process_file(file_path, source_label, csv_writer, handedness_data):
    """Process a single JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    for video_file, frame_data in data.items():
        try:
            session_id = video_file.split('_')[2]
            dominant_hand = determine_dominant_hand(session_id, handedness_data)

            # Get positions as numpy arrays
            left_pos = get_wrist_positions(frame_data, "Left")
            right_pos = get_wrist_positions(frame_data, "Right")

            # Ensure we're working with native Python integers
            total_frames = int(max(
                max(left_pos.keys(), default=0),
                max(right_pos.keys(), default=0)
            ))

            # Calculate metrics with type safety
            wrist_data = left_pos if dominant_hand == 'L' else right_pos
            metrics = compute_metrics(wrist_data, total_frames)

            # Convert all metrics to native Python types
            processed_metrics = [
                m.item() if isinstance(m, np.generic) else m
                for m in metrics
            ]

            if processed_metrics[0] is not None:
                csv_writer.writerow([
                    session_id,
                    source_label,
                    dominant_hand,
                    *processed_metrics
                ])

        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue


def main():
    """Main processing function."""
    # Load handedness data
    try:
        with open(SESSION_HANDEDNESS_PATH, 'r') as f:
            handedness_data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Error: Handedness file not found at {SESSION_HANDEDNESS_PATH}")

    # Create output CSV with pipeline-compatible headers
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'session_info_session_id',
            'adl_data_source',
            'adl_dominant_hand',
            'adl_face_washing_distance',
            'adl_face_washing_velocity',
            'adl_face_washing_velocity_abs',
            'adl_face_washing_acceleration',
            'adl_face_washing_acceleration_abs',
            'adl_face_washing_jerk',
            'adl_face_washing_jerk_abs',
            'adl_face_washing_used_frame_count',
            'adl_face_washing_percentage_frames_used'
        ])

        # Process both data sources
        for source_dir in SOURCE_DIRS:
            source_label = os.path.basename(source_dir)
            for root, _, files in os.walk(source_dir):
                if 'face' in root.lower():
                    for file in files:
                        if file.endswith('.json'):
                            process_file(
                                os.path.join(root, file),
                                source_label,
                                writer,
                                handedness_data
                            )


if __name__ == '__main__':
    main()
