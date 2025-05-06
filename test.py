import pandas as pd
merged_data_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv"
merged_df = pd.read_csv(merged_data_path)
# filtered_sessions = merged_df[merged_df['mediapipe_hand_face_percentage_frames_used_in_video'] >= 0.1]

print(merged_df[merged_df['mediapipe_hand_face_percentage_frames_used_in_video'] >= 0.1]['participant_id'].unique())
print(merged_df[merged_df['mediapipe_hand_face_percentage_frames_used_in_video'] < 0.1]['participant_id'].unique())