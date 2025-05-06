import pandas as pd

# Load the merged data and handedness annotation files
merged_data_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv"
handedness_data_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/handedness_annotations_20250313.csv"

# Load datasets
merged_df = pd.read_csv(merged_data_path)
handedness_df = pd.read_csv(handedness_data_path)

handedness_df['session_id'] = handedness_df['Recording_name'].str.split('_').str[2]
print(handedness_df['session_id'])

# Filter sessions where mediapipe_hand_face_percentage_frames_used_in_video < 0.1
filtered_sessions = merged_df[merged_df['mediapipe_hand_face_percentage_frames_used_in_video'] < 0.1]

# Extract session IDs from filtered sessions
filtered_session_ids = set(filtered_sessions['session_id'])

# Filter handedness data to include only relevant sessions
handedness_filtered = handedness_df

# Create sets for No_landmark_Drop and Bad_Session_Drop where value is 1
no_landmark_drop_sessions = set(
    handedness_filtered[handedness_filtered['No_landmark_Drop'] == 1]['session_id']
)
bad_session_drop_sessions = set(
    handedness_filtered[handedness_filtered['Bad_Session_Drop'] == 1]['session_id']
)

# Calculate Jaccard similarity for No_landmark_Drop sessions
no_landmark_intersection = no_landmark_drop_sessions & filtered_session_ids
no_landmark_union = no_landmark_drop_sessions | filtered_session_ids
no_landmark_jaccard_similarity = len(no_landmark_intersection) / len(no_landmark_union) if len(no_landmark_union) > 0 else 0

# Calculate Jaccard similarity for Bad_Session_Drop sessions
bad_session_intersection = bad_session_drop_sessions & filtered_session_ids
bad_session_union = bad_session_drop_sessions | filtered_session_ids
bad_session_jaccard_similarity = len(bad_session_intersection) / len(bad_session_union) if len(bad_session_union) > 0 else 0

# Output results
print(f"Number of sessions in No_landmark_Drop: {len(no_landmark_drop_sessions)}")
print(f"Number of overlapping sessions with filtered sessions (No_landmark_Drop): {len(no_landmark_intersection)}")
print(f"Jaccard Similarity for No_landmark_Drop: {no_landmark_jaccard_similarity:.2f}")

print(f"Number of sessions in Bad_Session_Drop: {len(bad_session_drop_sessions)}")
print(f"Number of overlapping sessions with filtered sessions (Bad_Session_Drop): {len(bad_session_intersection)}")
print(f"Jaccard Similarity for Bad_Session_Drop: {bad_session_jaccard_similarity:.2f}")
