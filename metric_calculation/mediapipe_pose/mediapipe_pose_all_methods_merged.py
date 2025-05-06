import os
import json
import pandas as pd
import numpy as np

# Load handedness annotations
handedness_annotations = pd.read_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/handedness_annotations_20250313.csv')

# Define directory path
directory_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_both_hands_entire_hand_nogbr'

# Define task folders
task_folders = {'face': [], 'teeth': [], 'hair': []}

# Identify folders for each task
for folder in os.listdir(directory_path):
    # print(folder)
    if 'face' in folder.lower() or "Face" in folder.lower():
        # print(folder)
        task_folders['face'].append(folder)
    elif 'teeth' or "Teeth" in folder.lower():
        task_folders['teeth'].append(folder)
    elif 'hair' or "Hair" in folder.lower():
        task_folders['hair'].append(folder)

# Initialize metrics storage
metrics_face = []
metrics_teeth = []
metrics_hair = []

# Utility Functions
def compute_velocity(positions):
    return np.diff(positions, axis=0)

def compute_acceleration(velocities):
    return np.gradient(velocities, axis=0) if len(velocities) >= 2 else None

def compute_jerk(accelerations):
    return np.gradient(accelerations, axis=0) if len(accelerations) >= 2 else None

def compute_mean_metrics(positions):
    if len(positions) < 2:  # Check if there are enough positions to compute metrics
        return None, None, None
    velocities = compute_velocity(positions)
    accelerations = compute_acceleration(velocities)
    jerks = compute_jerk(accelerations)
    if velocities is None or accelerations is None or jerks is None:
        return None, None, None
    return np.mean(np.linalg.norm(velocities, axis=1)), np.mean(np.linalg.norm(accelerations, axis=1)), np.mean(np.linalg.norm(jerks, axis=1))

def filter_frames(frames_data, condition):
    return [frame for frame, data in frames_data.items() if data and condition(data)]  # Skip empty dictionaries

# Filter recordings based on user options
def filter_recordings(option, handedness_annotations):
    if option == "no_exclusion":
        return handedness_annotations['Recording_name'].tolist()
    elif option == "no_landmark_drop":
        return handedness_annotations[handedness_annotations['No_landmark_Drop'] != 1]['Recording_name'].tolist()
    elif option == "bad_session_drop":
        return handedness_annotations[handedness_annotations['Bad_Session_Drop'] != 1]['Recording_name'].tolist()

# Iterate through task folders
for task, folders in task_folders.items():
    for folder in folders:
        folder_path = os.path.join(directory_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                recording_name = file.replace('.json', '.mp4')
                session_id = recording_name.split('_')[2]
                handedness_row = handedness_annotations[handedness_annotations['Recording_name'] == recording_name]

                # # Skip excluded recordings based on user choice
                # if handedness_row.empty:
                #     continue

                # Load JSON data
                with open(os.path.join(folder_path, file), 'r') as f:
                    json_data = json.load(f)

                # Extract frames and positions
                frames_data = json_data.get(list(json_data.keys())[0], {}).get(recording_name, {})

                # Task-specific processing: Face Task (9 Methods)
                if task == "face":
                    results = {'Recording_name': recording_name, 'Session_ID': session_id, 'Task_type': 'Face'}


                    #all columns added:
                    '''/*
                    columns = [
    # Method 1
    "both_hand_frames_mean_metric_velocity",
    "both_hand_frames_mean_metric_acceleration",
    "both_hand_frames_mean_metric_jerk",

    # Method 2
    "all_frames_mean_metric_velocity",
    "all_frames_mean_metric_acceleration",
    "all_frames_mean_metric_jerk",

    # Method 3
    "both_hand_frames_mean_landmark_velocity",
    "both_hand_frames_mean_landmark_acceleration",
    "both_hand_frames_mean_landmark_jerk",

    # Method 4
    "all_frames_mean_landmark_velocity",
    "all_frames_mean_landmark_acceleration",
    "all_frames_mean_landmark_jerk",

    # Method 5
    "both_hand_frames_mean_metric_centroid_velocity",
    "both_hand_frames_mean_metric_centroid_acceleration",
    "both_hand_frames_mean_metric_centroid_jerk",

    # Method 6
    "all_frames_mean_metric_centroid_velocity",
    "all_frames_mean_metric_centroid_acceleration",
    "all_frames_mean_metric_centroid_jerk",

    # Method 7
    "left_hand_velocity",
    "left_hand_acceleration",
    "left_hand_jerk",
    "right_hand_velocity",
    "right_hand_acceleration",
    "right_hand_jerk",
    "higher_velocity_hand_velocity",
    "higher_velocity_hand_acceleration",
    "higher_velocity_hand_jerk",
    "95_max_velocity",
    "95_max_acceleration",
    "95_max_jerk",
    "5_min_velocity",
    "5_min_acceleration",
    "5_min_jerk",

    # Method 8
    "mean_difference_velocity",
    "mean_difference_acceleration",
    "mean_difference_jerk"
]
                    */'''
                    # Method 1: Both hands detected (hand_status == 3)

                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            if frames_data[idx].get('hand_status') == 3 and frames_data[next_idx].get(
                                    'hand_status') == 3:
                                consecutive_frames.append(idx)

                    left_positions_consecutive = []
                    right_positions_consecutive = []

                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        pos_left_current = np.array(frames_data[idx]['hand_landmarks'].get('15', [0, 0]))
                        pos_right_current = np.array(frames_data[idx]['hand_landmarks'].get('16', [0, 0]))
                        pos_left_next = np.array(frames_data[next_idx]['hand_landmarks'].get('15', [0, 0]))
                        pos_right_next = np.array(frames_data[next_idx]['hand_landmarks'].get('16', [0, 0]))

                        left_positions_consecutive.append(pos_left_current)
                        left_positions_consecutive.append(pos_left_next)
                        right_positions_consecutive.append(pos_right_current)
                        right_positions_consecutive.append(pos_right_next)

                    left_metrics_consecutive = compute_mean_metrics(left_positions_consecutive)
                    right_metrics_consecutive = compute_mean_metrics(right_positions_consecutive)

                    results.update({
                        'both_hand_frames_mean_metric_velocity': (left_metrics_consecutive[0] +
                                                                  right_metrics_consecutive[0]) / 2 if
                        left_metrics_consecutive[0] is not None and right_metrics_consecutive[0] is not None else None,
                        'both_hand_frames_mean_metric_acceleration': (left_metrics_consecutive[1] +
                                                                      right_metrics_consecutive[1]) / 2 if
                        left_metrics_consecutive[1] is not None and right_metrics_consecutive[1] is not None else None,
                        'both_hand_frames_mean_metric_jerk': (left_metrics_consecutive[2] + right_metrics_consecutive[
                            2]) / 2 if left_metrics_consecutive[2] is not None and right_metrics_consecutive[
                            2] is not None else None,
                    })

                    # Method 2: At least one hand detected (hand_status != 0)
                    # Method 2: At least one hand detected consecutively (hand_status != 0)
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            if frames_data[idx].get('hand_status') and frames_data[next_idx].get(
                                    'hand_status') and frames_data[idx].get('hand_status') != 0 and frames_data[next_idx].get(
                                    'hand_status') != 0:
                                consecutive_frames.append(idx)

                    left_positions_consecutive = []
                    right_positions_consecutive = []

                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        pos_left_current = np.array(frames_data[idx]['hand_landmarks'].get('15', [0, 0]))
                        pos_right_current = np.array(frames_data[idx]['hand_landmarks'].get('16', [0, 0]))
                        pos_left_next = np.array(frames_data[next_idx]['hand_landmarks'].get('15', [0, 0]))
                        pos_right_next = np.array(frames_data[next_idx]['hand_landmarks'].get('16', [0, 0]))

                        left_positions_consecutive.append(pos_left_current)
                        left_positions_consecutive.append(pos_left_next)
                        right_positions_consecutive.append(pos_right_current)
                        right_positions_consecutive.append(pos_right_next)

                    left_metrics_consecutive = compute_mean_metrics(left_positions_consecutive)
                    right_metrics_consecutive = compute_mean_metrics(right_positions_consecutive)

                    results.update({
                        'all_frames_mean_metric_velocity': (left_metrics_consecutive[0] + right_metrics_consecutive[
                            0]) / 2 if left_metrics_consecutive[0] is not None and right_metrics_consecutive[
                            0] is not None else None,
                        'all_frames_mean_metric_acceleration': (left_metrics_consecutive[1] + right_metrics_consecutive[
                            1]) / 2 if left_metrics_consecutive[1] is not None and right_metrics_consecutive[
                            1] is not None else None,
                        'all_frames_mean_metric_jerk': (left_metrics_consecutive[2] + right_metrics_consecutive[
                            2]) / 2 if left_metrics_consecutive[2] is not None and right_metrics_consecutive[
                            2] is not None else None,
                    })

                    # Method 3: Both hands detected consecutively (hand_status == 3)
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            # Ensure both frames meet the condition for hand_status == 3
                            if frames_data[idx].get('hand_status') == 3 and frames_data[next_idx].get(
                                    'hand_status') == 3:
                                consecutive_frames.append(idx)

                    # Initialize lists to store metrics
                    velocity_list = []
                    acceleration_list = []
                    jerk_list = []

                    # Iterate through consecutive frames to calculate metrics
                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        if frames_data[idx] and frames_data[next_idx]:
                            # Extract positions for current and next frames
                            pos_left_current = np.array(frames_data[idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_current = np.array(frames_data[idx]['hand_landmarks'].get('16', [0, 0]))
                            pos_left_next = np.array(frames_data[next_idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_next = np.array(frames_data[next_idx]['hand_landmarks'].get('16', [0, 0]))

                            # Calculate mean landmark position
                            mean_pos_current = (pos_left_current + pos_right_current) / 2
                            mean_pos_next = (pos_left_next + pos_right_next) / 2

                            # Calculate velocity
                            velocity = mean_pos_next - mean_pos_current
                            velocity_list.append(velocity)

                    # Calculate acceleration and jerk if there are enough data points
                    if len(velocity_list) > 1:
                        velocities_array = np.array(velocity_list)
                        accelerations = compute_acceleration(velocities_array)
                        if accelerations is not None:
                            acceleration_list.extend(accelerations)

                    if len(acceleration_list) > 1:
                        accelerations_array = np.array(acceleration_list)
                        jerks = compute_jerk(accelerations_array)
                        if jerks is not None:
                            jerk_list.extend(jerks)

                    # Calculate average metrics
                    average_velocity = np.mean(
                        np.linalg.norm(np.array(velocity_list), axis=1)) if velocity_list else None
                    average_acceleration = np.mean(
                        np.linalg.norm(np.array(acceleration_list), axis=1)) if acceleration_list else None
                    average_jerk = np.mean(np.linalg.norm(np.array(jerk_list), axis=1)) if jerk_list else None

                    # Update results with average metrics
                    results.update({
                        'both_hand_frames_mean_landmark_velocity': average_velocity,
                        'both_hand_frames_mean_landmark_acceleration': average_acceleration,
                        'both_hand_frames_mean_landmark_jerk': average_jerk,
                    })
                    # Method 4: At least one hand detected consecutively (hand_status != 0)
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            # Ensure both frames meet the condition for hand_status != 0
                            if frames_data[idx].get('hand_status') and frames_data[next_idx].get('hand_status') and \
                                    frames_data[idx].get('hand_status') != 0 and frames_data[next_idx].get(
                                'hand_status') != 0:
                                consecutive_frames.append(idx)

                    # Initialize lists to store metrics
                    velocity_list = []
                    acceleration_list = []
                    jerk_list = []

                    # Iterate through consecutive frames to calculate metrics
                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        if frames_data[idx] and frames_data[next_idx]:
                            # Extract positions for current and next frames
                            pos_left_current = np.array(frames_data[idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_current = np.array(frames_data[idx]['hand_landmarks'].get('16', [0, 0]))
                            pos_left_next = np.array(frames_data[next_idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_next = np.array(frames_data[next_idx]['hand_landmarks'].get('16', [0, 0]))

                            # Calculate mean landmark position
                            mean_pos_current = (pos_left_current + pos_right_current) / 2
                            mean_pos_next = (pos_left_next + pos_right_next) / 2

                            # Calculate velocity
                            velocity = mean_pos_next - mean_pos_current
                            velocity_list.append(velocity)

                    # Calculate acceleration and jerk if there are enough data points
                    if len(velocity_list) > 1:
                        accelerations = compute_acceleration(np.array(velocity_list))
                        jerks = compute_jerk(accelerations)
                        acceleration_list.extend(accelerations)
                        jerk_list.extend(jerks)

                    # Calculate average metrics
                    average_velocity = np.mean(
                        np.linalg.norm(np.array(velocity_list), axis=1)) if velocity_list else None
                    average_acceleration = np.mean(
                        np.linalg.norm(np.array(acceleration_list), axis=1)) if acceleration_list else None
                    average_jerk = np.mean(np.linalg.norm(np.array(jerk_list), axis=1)) if jerk_list else None

                    # Update results with average metrics
                    results.update({
                        'all_frames_mean_landmark_velocity': average_velocity,
                        'all_frames_mean_landmark_acceleration': average_acceleration,
                        'all_frames_mean_landmark_jerk': average_jerk,
                    })
                    # Method 5: Both hands detected consecutively (hand_status == 3) using centroid positions
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            # Ensure both frames meet the condition for hand_status == 3
                            if frames_data[idx].get('hand_status') == 3 and frames_data[next_idx].get(
                                    'hand_status') == 3:
                                consecutive_frames.append(idx)

                    # Initialize lists to store metrics for left and right hands
                    left_velocity_list = []
                    right_velocity_list = []
                    left_acceleration_list = []
                    right_acceleration_list = []
                    left_jerk_list = []
                    right_jerk_list = []

                    # Iterate through consecutive frames to calculate metrics
                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        if frames_data[idx] and frames_data[next_idx]:
                            # Extract positions for current and next frames
                            left_landmarks_current = [np.array(frames_data[idx]['hand_landmarks'].get(str(i), [0, 0]))
                                                      for i in [15, 17, 19, 21]]
                            right_landmarks_current = [np.array(frames_data[idx]['hand_landmarks'].get(str(i), [0, 0]))
                                                       for i in [16, 18, 20, 22]]
                            left_landmarks_next = [np.array(frames_data[next_idx]['hand_landmarks'].get(str(i), [0, 0]))
                                                   for i in [15, 17, 19, 21]]
                            right_landmarks_next = [
                                np.array(frames_data[next_idx]['hand_landmarks'].get(str(i), [0, 0])) for i in
                                [16, 18, 20, 22]]

                            # Calculate centroids for current and next frames
                            left_centroid_current = np.mean(left_landmarks_current, axis=0)
                            right_centroid_current = np.mean(right_landmarks_current, axis=0)
                            left_centroid_next = np.mean(left_landmarks_next, axis=0)
                            right_centroid_next = np.mean(right_landmarks_next, axis=0)

                            # Calculate velocity for left and right hands
                            left_velocity = left_centroid_next - left_centroid_current
                            right_velocity = right_centroid_next - right_centroid_current

                            left_velocity_list.append(left_velocity)
                            right_velocity_list.append(right_velocity)

                    # Calculate acceleration and jerk if there are enough data points
                    if len(left_velocity_list) > 1:
                        left_velocities_array = np.array(left_velocity_list)
                        left_accelerations = compute_acceleration(left_velocities_array)
                        if left_accelerations is not None:
                            left_acceleration_list.extend(left_accelerations)

                    if len(right_velocity_list) > 1:
                        right_velocities_array = np.array(right_velocity_list)
                        right_accelerations = compute_acceleration(right_velocities_array)
                        if right_accelerations is not None:
                            right_acceleration_list.extend(right_accelerations)

                    if len(left_acceleration_list) > 1:
                        left_accelerations_array = np.array(left_acceleration_list)
                        left_jerks = compute_jerk(left_accelerations_array)
                        if left_jerks is not None:
                            left_jerk_list.extend(left_jerks)

                    if len(right_acceleration_list) > 1:
                        right_accelerations_array = np.array(right_acceleration_list)
                        right_jerks = compute_jerk(right_accelerations_array)
                        if right_jerks is not None:
                            right_jerk_list.extend(right_jerks)

                    # Calculate average metrics for both hands
                    average_left_velocity = np.mean(
                        np.linalg.norm(np.array(left_velocity_list), axis=1)) if left_velocity_list else None
                    average_right_velocity = np.mean(
                        np.linalg.norm(np.array(right_velocity_list), axis=1)) if right_velocity_list else None

                    average_left_acceleration = np.mean(
                        np.linalg.norm(np.array(left_acceleration_list), axis=1)) if left_acceleration_list else None
                    average_right_acceleration = np.mean(
                        np.linalg.norm(np.array(right_acceleration_list), axis=1)) if right_acceleration_list else None

                    average_left_jerk = np.mean(
                        np.linalg.norm(np.array(left_jerk_list), axis=1)) if left_jerk_list else None
                    average_right_jerk = np.mean(
                        np.linalg.norm(np.array(right_jerk_list), axis=1)) if right_jerk_list else None

                    # Calculate overall averages across both hands
                    average_centroid_velocity = (
                                                            average_left_velocity + average_right_velocity) / 2 if average_left_velocity is not None and average_right_velocity is not None else None
                    average_centroid_acceleration = (
                                                                average_left_acceleration + average_right_acceleration) / 2 if average_left_acceleration is not None and average_right_acceleration is not None else None
                    average_centroid_jerk = (
                                                        average_left_jerk + average_right_jerk) / 2 if average_left_jerk is not None and average_right_jerk is not None else None

                    # Update results with average metrics
                    results.update({
                        'both_hand_frames_mean_metric_centroid_velocity': average_centroid_velocity,
                        'both_hand_frames_mean_metric_centroid_acceleration': average_centroid_acceleration,
                        'both_hand_frames_mean_metric_centroid_jerk': average_centroid_jerk,
                    })

                    # Method 6: At least one hand detected consecutively (hand_status != 0) using centroid positions
                    import numpy as np

                    # Step 1: Filter consecutive frames where at least one hand is detected
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            # Ensure both frames meet the condition for hand_status != 0
                            if frames_data[idx].get('hand_status') and frames_data[next_idx].get('hand_status') and \
                                    frames_data[idx].get('hand_status') != 0 and frames_data[next_idx].get(
                                'hand_status') != 0:
                                consecutive_frames.append(idx)

                    # Step 2: Calculate centroids for left and right hands
                    left_centroids = []
                    right_centroids = []

                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        if frames_data[idx] and frames_data[next_idx]:
                            # Extract positions for current and next frames
                            left_landmarks_current = [
                                np.array(frames_data[idx]['hand_landmarks'].get(str(i), [0, 0])) for i in
                                [15, 17, 19, 21]
                            ]
                            right_landmarks_current = [
                                np.array(frames_data[idx]['hand_landmarks'].get(str(i), [0, 0])) for i in
                                [16, 18, 20, 22]
                            ]

                            left_landmarks_next = [
                                np.array(frames_data[next_idx]['hand_landmarks'].get(str(i), [0, 0])) for i in
                                [15, 17, 19, 21]
                            ]
                            right_landmarks_next = [
                                np.array(frames_data[next_idx]['hand_landmarks'].get(str(i), [0, 0])) for i in
                                [16, 18, 20, 22]
                            ]

                            # Calculate centroids for current and next frames
                            left_centroid_current = np.mean(left_landmarks_current, axis=0)
                            right_centroid_current = np.mean(right_landmarks_current, axis=0)
                            left_centroid_next = np.mean(left_landmarks_next, axis=0)
                            right_centroid_next = np.mean(right_landmarks_next, axis=0)

                            left_centroids.append((left_centroid_current, left_centroid_next))
                            right_centroids.append((right_centroid_current, right_centroid_next))


                    # Step 3: Calculate velocity, acceleration, and jerk metrics
                    def calculate_metrics(centroids):
                        velocities = []
                        accelerations = []
                        jerks = []

                        for current, next in centroids:
                            velocity = next - current
                            velocities.append(velocity)

                        if len(velocities) > 1:
                            velocities_array = np.array(velocities)
                            accelerations = np.gradient(velocities_array, axis=0)

                        if len(accelerations) > 1:
                            accelerations_array = np.array(accelerations)
                            jerks = np.gradient(accelerations_array, axis=0)

                        # Ensure that velocities, accelerations, and jerks have valid data before calculating averages
                        average_velocity = (
                            np.mean(np.linalg.norm(np.array(velocities), axis=1)) if len(velocities) > 0 else None
                        )
                        average_acceleration = (
                            np.mean(np.linalg.norm(np.array(accelerations), axis=1)) if len(accelerations) > 0 else None
                        )
                        average_jerk = (
                            np.mean(np.linalg.norm(np.array(jerks), axis=1)) if len(jerks) > 0 else None
                        )

                        return average_velocity, average_acceleration, average_jerk


                    # Compute metrics for left and right hands
                    avg_left_velocity, avg_left_acceleration, avg_left_jerk = calculate_metrics(left_centroids)
                    avg_right_velocity, avg_right_acceleration, avg_right_jerk = calculate_metrics(right_centroids)

                    # Step 4: Aggregate results across both hands
                    average_velocity = (
                                                   avg_left_velocity + avg_right_velocity) / 2 if avg_left_velocity is not None and avg_right_velocity is not None else None
                    average_acceleration = (
                                                       avg_left_acceleration + avg_right_acceleration) / 2 if avg_left_acceleration is not None and avg_right_acceleration is not None else None
                    average_jerk = (
                                               avg_left_jerk + avg_right_jerk) / 2 if avg_left_jerk is not None and avg_right_jerk is not None else None

                    # Step 5: Update results with average metrics
                    results.update({
                        'all_frames_mean_metric_centroid_velocity': average_velocity,
                        'all_frames_mean_metric_centroid_acceleration': average_acceleration,
                        'all_frames_mean_metric_centroid_jerk': average_jerk,
                    })

                    # Method 7: At least one hand detected consecutively (hand_status != 0) using wrist landmarks
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            # Ensure both frames meet the condition for hand_status != 0
                            if frames_data[idx].get('hand_status') and frames_data[next_idx].get('hand_status') and \
                                    frames_data[idx].get('hand_status') != 0 and frames_data[next_idx].get(
                                'hand_status') != 0:
                                consecutive_frames.append(idx)

                    # Initialize lists to store metrics for left and right hands
                    left_velocity_list = []
                    right_velocity_list = []
                    left_acceleration_list = []
                    right_acceleration_list = []
                    left_jerk_list = []
                    right_jerk_list = []

                    # Iterate through consecutive frames to calculate metrics
                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        if frames_data[idx] and frames_data[next_idx]:
                            # Extract wrist positions for current and next frames
                            pos_left_current = np.array(frames_data[idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_current = np.array(frames_data[idx]['hand_landmarks'].get('16', [0, 0]))
                            pos_left_next = np.array(frames_data[next_idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_next = np.array(frames_data[next_idx]['hand_landmarks'].get('16', [0, 0]))

                            # Calculate velocity for left and right hands
                            left_velocity = pos_left_next - pos_left_current
                            right_velocity = pos_right_next - pos_right_current

                            left_velocity_list.append(left_velocity)
                            right_velocity_list.append(right_velocity)

                    # Calculate acceleration and jerk if there are enough data points
                    if len(left_velocity_list) > 1:
                        left_velocities_array = np.array(left_velocity_list)
                        left_accelerations = compute_acceleration(left_velocities_array)
                        if left_accelerations is not None:
                            left_acceleration_list.extend(left_accelerations)

                    if len(right_velocity_list) > 1:
                        right_velocities_array = np.array(right_velocity_list)
                        right_accelerations = compute_acceleration(right_velocities_array)
                        if right_accelerations is not None:
                            right_acceleration_list.extend(right_accelerations)

                    if len(left_acceleration_list) > 1:
                        left_accelerations_array = np.array(left_acceleration_list)
                        left_jerks = compute_jerk(left_accelerations_array)
                        if left_jerks is not None:
                            left_jerk_list.extend(left_jerks)

                    if len(right_acceleration_list) > 1:
                        right_accelerations_array = np.array(right_acceleration_list)
                        right_jerks = compute_jerk(right_accelerations_array)
                        if right_jerks is not None:
                            right_jerk_list.extend(right_jerks)

                    # Calculate average metrics for both hands
                    average_left_velocity = np.mean(np.linalg.norm(np.array(left_velocity_list), axis=1)) if len(
                        left_velocity_list) > 0 else None
                    average_right_velocity = np.mean(np.linalg.norm(np.array(right_velocity_list), axis=1)) if len(
                        right_velocity_list) > 0 else None

                    average_left_acceleration = np.mean(
                        np.linalg.norm(np.array(left_acceleration_list), axis=1)) if len(
                        left_acceleration_list) > 0 else None
                    average_right_acceleration = np.mean(
                        np.linalg.norm(np.array(right_acceleration_list), axis=1)) if len(
                        right_acceleration_list) > 0 else None

                    average_left_jerk = np.mean(np.linalg.norm(np.array(left_jerk_list), axis=1)) if len(
                        left_jerk_list) > 0 else None
                    average_right_jerk = np.mean(np.linalg.norm(np.array(right_jerk_list), axis=1)) if len(
                        right_jerk_list) > 0 else None

                    # Determine higher velocity hand metrics
                    if average_left_velocity is not None and average_right_velocity is not None:
                        if average_left_velocity >= average_right_velocity:
                            higher_hand_metrics = (average_left_velocity, average_left_acceleration, average_left_jerk)
                        else:
                            higher_hand_metrics = (
                            average_right_velocity, average_right_acceleration, average_right_jerk)
                    else:
                        higher_hand_metrics = (None, None, None)

                    # Pool all metrics from both hands together for global statistics
                    pooled_velocities_magnitudes = np.linalg.norm(np.array(left_velocity_list + right_velocity_list),
                                                                  axis=1) if len(
                        left_velocity_list + right_velocity_list) > 0 else []
                    pooled_accelerations_magnitudes = np.linalg.norm(
                        np.array(left_acceleration_list + right_acceleration_list), axis=1) if len(
                        left_acceleration_list + right_acceleration_list) > 0 else []
                    pooled_jerks_magnitudes = np.linalg.norm(np.array(left_jerk_list + right_jerk_list), axis=1) if len(
                        left_jerk_list + right_jerk_list) > 0 else []

                    percentile_95_max_velocity = np.percentile(pooled_velocities_magnitudes, 95) if len(
                        pooled_velocities_magnitudes) > 0 else None
                    percentile_95_max_acceleration = np.percentile(pooled_accelerations_magnitudes, 95) if len(
                        pooled_accelerations_magnitudes) > 0 else None
                    percentile_95_max_jerk = np.percentile(pooled_jerks_magnitudes, 95) if len(
                        pooled_jerks_magnitudes) > 0 else None

                    percentile_5_min_velocity = np.percentile(pooled_velocities_magnitudes, 5) if len(
                        pooled_velocities_magnitudes) > 0 else None
                    percentile_5_min_acceleration = np.percentile(pooled_accelerations_magnitudes, 5) if len(
                        pooled_accelerations_magnitudes) > 0 else None
                    percentile_5_min_jerk = np.percentile(pooled_jerks_magnitudes, 5) if len(
                        pooled_jerks_magnitudes) > 0 else None

                    # Update results with calculated metrics
                    results.update({
                        'left_hand_velocity': average_left_velocity,
                        'left_hand_acceleration': average_left_acceleration,
                        'left_hand_jerk': average_left_jerk,
                        'right_hand_velocity': average_right_velocity,
                        'right_hand_acceleration': average_right_acceleration,
                        'right_hand_jerk': average_right_jerk,
                        'higher_velocity_hand_velocity': higher_hand_metrics[0],
                        'higher_velocity_hand_acceleration': higher_hand_metrics[1],
                        'higher_velocity_hand_jerk': higher_hand_metrics[2],
                        '95_max_velocity': percentile_95_max_velocity,
                        '95_max_acceleration': percentile_95_max_acceleration,
                        '95_max_jerk': percentile_95_max_jerk,
                        '5_min_velocity': percentile_5_min_velocity,
                        '5_min_acceleration': percentile_5_min_acceleration,
                        '5_min_jerk': percentile_5_min_jerk,
                    })

                    # Method 8: Both hands detected consecutively (hand_status == 3) using wrist landmarks
                    import numpy as np

                    # Step 1: Filter consecutive frames where both hands are detected
                    consecutive_frames = []
                    sorted_keys = sorted(frames_data.keys(), key=int)

                    for idx in sorted_keys[:-1]:  # Exclude the last frame to compare with the next one
                        next_idx = str(int(idx) + 1)
                        if idx in frames_data and next_idx in frames_data:
                            # Ensure both frames meet the condition for hand_status == 3
                            if frames_data[idx].get('hand_status') == 3 and frames_data[next_idx].get(
                                    'hand_status') == 3:
                                consecutive_frames.append(idx)

                    # Step 2: Calculate wrist positions for left and right hands
                    left_positions = []
                    right_positions = []

                    for idx in consecutive_frames:
                        next_idx = str(int(idx) + 1)
                        if frames_data[idx] and frames_data[next_idx]:
                            # Extract wrist positions for current and next frames
                            pos_left_current = np.array(frames_data[idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_current = np.array(frames_data[idx]['hand_landmarks'].get('16', [0, 0]))
                            pos_left_next = np.array(frames_data[next_idx]['hand_landmarks'].get('15', [0, 0]))
                            pos_right_next = np.array(frames_data[next_idx]['hand_landmarks'].get('16', [0, 0]))

                            # Append positions for velocity calculation
                            left_positions.append((pos_left_current, pos_left_next))
                            right_positions.append((pos_right_current, pos_right_next))


                    # Step 3: Calculate velocity, acceleration, and jerk metrics
                    def calculate_metrics(positions):
                        velocities = []
                        accelerations = []
                        jerks = []

                        for current, next in positions:
                            velocity = next - current
                            velocities.append(velocity)

                        if len(velocities) > 1:
                            velocities_array = np.array(velocities)
                            accelerations = np.gradient(velocities_array, axis=0)

                        if len(accelerations) > 1:
                            accelerations_array = np.array(accelerations)
                            jerks = np.gradient(accelerations_array, axis=0)

                        # Calculate average metrics
                        average_velocity = np.mean(np.linalg.norm(np.array(velocities), axis=1)) if len(
                            velocities) > 0 else None
                        average_acceleration = np.mean(np.linalg.norm(np.array(accelerations), axis=1)) if len(
                            accelerations) > 0 else None
                        average_jerk = np.mean(np.linalg.norm(np.array(jerks), axis=1)) if len(jerks) > 0 else None

                        return velocities, accelerations, jerks, average_velocity, average_acceleration, average_jerk


                    # Compute metrics for left and right hands
                    left_velocities, left_accelerations, left_jerks, avg_left_velocity, avg_left_acceleration, avg_left_jerk = calculate_metrics(
                        left_positions)
                    right_velocities, right_accelerations, right_jerks, avg_right_velocity, avg_right_acceleration, avg_right_jerk = calculate_metrics(
                        right_positions)

                    # Step 4: Calculate absolute differences between metrics of left and right hands
                    abs_diff_velocities = [np.abs(np.linalg.norm(lv) - np.linalg.norm(rv)) for lv, rv in
                                           zip(left_velocities, right_velocities)]
                    abs_diff_accelerations = [np.abs(np.linalg.norm(la) - np.linalg.norm(ra)) for la, ra in
                                              zip(left_accelerations, right_accelerations)]
                    abs_diff_jerks = [np.abs(np.linalg.norm(lj) - np.linalg.norm(rj)) for lj, rj in
                                      zip(left_jerks, right_jerks)]

                    # Step 5: Calculate mean differences across all selected frames
                    mean_difference_velocity = np.mean(abs_diff_velocities) if abs_diff_velocities else None
                    mean_difference_acceleration = np.mean(abs_diff_accelerations) if abs_diff_accelerations else None
                    mean_difference_jerk = np.mean(abs_diff_jerks) if abs_diff_jerks else None

                    # Step 6: Update results with calculated metrics
                    results.update({
                        'mean_difference_velocity': mean_difference_velocity,
                        'mean_difference_acceleration': mean_difference_acceleration,
                        'mean_difference_jerk': mean_difference_jerk,
                    })

                    metrics_face.append(results)

                # Task-specific processing: Teeth and Hair Tasks
                elif task in ["teeth", "hair"]:
                    if handedness_row.empty:
                    # continue
                        handedness_value = 'Both'
                    else:
                        handedness_value = handedness_row.iloc[0]['Handedness']

                    valid_frames = filter_frames(frames_data, lambda d: d.get('hand_status') != 0)

                    left_positions_teeth_hair = [frames_data[frame]['hand_landmarks'].get('15', [0, 0]) for frame in valid_frames]
                    right_positions_teeth_hair = [frames_data[frame]['hand_landmarks'].get('16', [0, 0]) for frame in valid_frames]

                    velocity_left, acceleration_left, jerk_left = compute_mean_metrics(left_positions_teeth_hair)
                    velocity_right, acceleration_right, jerk_right = compute_mean_metrics(right_positions_teeth_hair)

                    if handedness_value == "L":
                        velocity, acceleration, jerk = velocity_left, acceleration_left, jerk_left
                    elif handedness_value == "R":
                        velocity, acceleration, jerk = velocity_right, acceleration_right, jerk_right
                    else:
                        velocity = (velocity_left + velocity_right) / 2 if velocity_left is not None and velocity_right is not None else None
                        acceleration = (acceleration_left + acceleration_right) / 2 if acceleration_left is not None and acceleration_right is not None else None
                        jerk = (jerk_left + jerk_right) / 2 if jerk_left is not None and jerk_right is not None else None

                    metrics_teeth.append({
                        'Recording_name': recording_name,
                        'Session_ID': session_id,
                        'Task_type': task.capitalize(),
                        'velocity': velocity,
                        'acceleration': acceleration,
                        'jerk': jerk,
                    })

# Save metrics to CSV files
metrics_face_df = pd.DataFrame(metrics_face)
metrics_face_df.to_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/merged_methods/metrics_face.csv', index=False)

metrics_teeth_df = pd.DataFrame(metrics_teeth)
metrics_teeth_df.to_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/merged_methods/metrics_teeth.csv', index=False)

metrics_hair_df = pd.DataFrame(metrics_hair)
metrics_hair_df.to_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_nobgr_results/merged_methods/metrics_hair.csv', index=False)
