

import csv
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def load_survey_scores(survey_json_path):
    """Load survey scores from JSON file and extract relevant questions."""
    with open(survey_json_path, 'r') as f:
        survey_data = json.load(f)

    extracted_scores = {}
    for session, scores in survey_data.items():
        if all(q in scores for q in ["5_1", "5_2", "6", "7"]):
            extracted_scores[session] = {
                "5_1": scores["5_1"],
                "5_2": scores["5_2"],
                "6": scores["6"],
                "7": scores["7"]
            }
    return extracted_scores


def compute_correlations(df, output_dir):
    """Compute correlations and store in CSV files."""
    # metrics = ["Velocity", "Acceleration", "Jerk","Higher_Velocity","Higher_Acceleration","Higher_Jerk"]
    # metrics = ["Velocity", "Acceleration", "Jerk","Left Mean Velocity",
    #         "Right Mean Velocity",
    #         "95th Percentile Velocity",
    #         "5th Percentile Velocity",
    #         "Mean of All Means Velocity",
    #         "Mean Difference Velocity"]
    # metrics = ["Distance","Velocity", "Acceleration", "Jerk"]
    # metrics = ["Velocity", "Acceleration", "Jerk"]
    metrics = [

        "Distance",
        "Distance Over Time",
        "Velocity",
        "Acceleration",
        "Jerk",
        "Number of Frames Used",
        "Total Frames",
        "Percentage of Frames Used"
    ]
    # metrics = [
    #
    #         "Distance Left Hand",
    #         "Velocity Left Hand",
    #         "Acceleration Left Hand",
    #         "Jerk Left Hand",
    #         "Distance Right Hand",
    #         "Velocity Right Hand",
    #         "Acceleration Right Hand",
    #         "Jerk Right Hand",
    #         "Distance (Higher Velocity)",
    #         "Velocity (Higher Velocity)",
    #         "Acceleration (Higher Velocity Hand)",
    #         "Jerk (Higher Velocity Hand)",
    #         "Average Distance (Both Hands)",
    #         "Average Velocity (Both Hands)",
    #         "Average Acceleration (Both Hands)",
    #         "Average Jerk (Both Hands)"
    #     ]

    # metrics = [
    #         "Velocity Left Hand",
    #         "Acceleration Left Hand",
    #         "Jerk Left Hand",
    #         "Velocity Right Hand",
    #         "Acceleration Right Hand",
    #         "Jerk Right Hand",
    #         "Velocity (Higher Velocity)",
    #         "Acceleration (Higher Velocity Hand)",
    #         "Jerk (Higher Velocity Hand)",
    #         "Average Velocity (Both Hands)",
    #         "Average Acceleration (Both Hands)",
    #         "Average Jerk (Both Hands)"
    #     ]


    # metrics = [
    #     # Method 1
    #     "both_hand_frames_mean_metric_velocity",
    #     "both_hand_frames_mean_metric_acceleration",
    #     "both_hand_frames_mean_metric_jerk",
    #
    #     # Method 2
    #     "all_frames_mean_metric_velocity",
    #     "all_frames_mean_metric_acceleration",
    #     "all_frames_mean_metric_jerk",
    #
    #     # Method 3
    #     "both_hand_frames_mean_landmark_velocity",
    #     "both_hand_frames_mean_landmark_acceleration",
    #     "both_hand_frames_mean_landmark_jerk",
    #
    #     # Method 4
    #     "all_frames_mean_landmark_velocity",
    #     "all_frames_mean_landmark_acceleration",
    #     "all_frames_mean_landmark_jerk",
    #
    #     # Method 5
    #     "both_hand_frames_mean_metric_centroid_velocity",
    #     "both_hand_frames_mean_metric_centroid_acceleration",
    #     "both_hand_frames_mean_metric_centroid_jerk",
    #
    #     # Method 6
    #     "all_frames_mean_metric_centroid_velocity",
    #     "all_frames_mean_metric_centroid_acceleration",
    #     "all_frames_mean_metric_centroid_jerk",
    #
    #     # Method 7
    #     "left_hand_velocity",
    #     "left_hand_acceleration",
    #     "left_hand_jerk",
    #     "right_hand_velocity",
    #     "right_hand_acceleration",
    #     "right_hand_jerk",
    #     "higher_velocity_hand_velocity",
    #     "higher_velocity_hand_acceleration",
    #     "higher_velocity_hand_jerk",
    #     "95_max_velocity",
    #     "95_max_acceleration",
    #     "95_max_jerk",
    #     "5_min_velocity",
    #     "5_min_acceleration",
    #     "5_min_jerk",
    #
    #     # Method 8
    #     "mean_difference_velocity",
    #     "mean_difference_acceleration",
    #     "mean_difference_jerk"
    # ]
    # metrics = ["Average Velocity (Both Hands)", "Average Acceleration (Both Hands)","Average Jerk (Both Hands)"]
    # metrics = ["Velocity (Higher Velocity)", "Acceleration (Higher Velocity Hand)","Jerk (Higher Velocity Hand)"]
    # metrics = ["Velocity", "Acceleration", "Jerk", "Left Mean Velocity", "Right Mean Velocity", "95th Percentile Velocity", "5th Percentile Velocity", "Mean of All Means Velocity", "Mean Difference Velocity"]
    # metrics = ["Velocity (Dominant Hand)", "Acceleration (Dominant Hand)", "Jerk (Dominant Hand)"]

    survey_questions = ["5_1", "5_2", "6", "7"]


    print(df.head())
    # df = df[df['Percentage Frames Used in Video'] >= 0.1]
    correlation_results = {}

    for question in survey_questions:
        results = []

        for metric in metrics:
            valid_data = df[[question, metric]].dropna()
            if not valid_data.empty:
                pearson_corr, pearson_p = pearsonr(valid_data[question], valid_data[metric])
                spearman_corr, spearman_p = spearmanr(valid_data[question], valid_data[metric], nan_policy='omit')
            else:
                pearson_corr, pearson_p, spearman_corr, spearman_p = np.nan, np.nan, np.nan, np.nan

            results.append([metric, pearson_corr, pearson_p, spearman_corr, spearman_p])
            correlation_results.setdefault(question, []).append(
                {"Metric": metric, "Pearson Corr": pearson_corr, "Pearson p-value": pearson_p,
                 "Spearman Corr": spearman_corr, "Spearman p-value": spearman_p})

        # Save individual survey question correlation results
        # question_df = pd.DataFrame(results, columns=["Metric", "Pearson Corr", "Pearson p-value", "Spearman Corr",
        #                                              "Spearman p-value"])
        # # question_df.to_csv(f"{output_dir}/correlation_{question}.csv", index=False)

    # Save overall correlation results
    correlation_df = pd.concat(
        [pd.DataFrame(v).assign(Question=k) for k, v in correlation_results.items()], ignore_index=True)
    correlation_df.to_csv(f"{output_dir}/correlation_dominant_hand_metrics_face.csv", index=False)


def merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, cohort_csv):
    """Merge the original CSV with survey scores and compute correlation."""
    df = pd.read_csv(original_csv)
    cohort_df = pd.read_csv(cohort_csv)
    # df = df[df['JSON File Name'] != "2024-11-04-19-25-54_7030_64ffb93c-54f1-4631-8f54-f609962d4dc4_patientTurn.json"]
    # df = df[df['Percentage Frames Used in Video (Both Hands)'] >= 0.1]
    # df = df[df['Percentage Frames Used in Video (Both Hands)'] >= 0.1]
    # Merge survey scores into DataFrame
    df = df.merge(pd.DataFrame.from_dict(survey_scores, orient="index"),
                  left_on="Session_ID", right_index=True, how="left")
    print(df.shape)

    df = df.merge(cohort_df, left_on="Session_ID", right_on="Session ID", how="left")
    print(df.shape)

    df = df[df['Cohort'] == 'patient']

    # Convert survey scores to numeric (handle "no" as NaN)
    df[["5_1", "5_2", "6", "7"]] = df[["5_1", "5_2", "6", "7"]].apply(pd.to_numeric, errors='coerce')

    # df = df.dropna()
    print(df.shape)
    # Save updated CSV
    df.to_csv(output_csv, index=False)

    # Compute and save correlation results
    compute_correlations(df, output_dir)


if __name__ == "__main__":
    # survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"  # Replace with actual path
    # original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/dominant_hand/dominant_hand_metrics_face.csv"  # CSV from previous script
    # output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/dominant_hand/merged_output_v2.csv"
    # output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/dominant_hand/"  # Replace with actual path
    # cohort_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"

    # survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"  # Replace with actual path
    # original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/dominant_hand/dominant_hand_metrics_hair.csv"  # CSV from previous script
    # output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/dominant_hand/merged_output_v2.csv"
    # output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/dominant_hand/"  # Replace with actual path
    # cohort_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"

    # survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"  # Replace with actual path
    # original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/metrics_face.csv"  # CSV from previous script
    # output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/merged_output_v2.csv"
    # output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/"  # Replace with actual path
    # cohort_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"

    # survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"  # Replace with actual path
    # original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/centroid/metrics_face.csv"  # CSV from previous script
    # output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/merged_output_v2.csv"
    # output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL//mediapipe_hand_nobgr_results/centroid/"  # Replace with actual path
    # cohort_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"


    survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"  # Replace with actual path
    original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/dominant_hand_metrics_face.csv"  # CSV from previous script
    output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/merged_output_v2.csv"
    output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/"  # Replace with actual path
    cohort_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"


    # survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json"  # Replace with actual path
    # original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/dominant_hand_metrics_hair.csv"  # CSV from previous script
    # output_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/merged_output_v2.csv"
    # output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_hand_nobgr_results/abs/"  # Replace with actual path
    # cohort_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv"


    survey_scores = load_survey_scores(survey_json_path)
    merge_csv_with_survey(original_csv, survey_scores, output_csv, output_dir, cohort_csv)
