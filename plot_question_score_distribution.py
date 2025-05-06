import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_survey_score_distribution(survey_scores, original_csv):
    """Plot the distribution of the survey question scores for each survey question."""
    # Load the original CSV
    df = pd.read_csv(original_csv)

    # Filter out sessions not present in both survey data and original CSV
    valid_sessions = df['Session Name'].isin(survey_scores.keys())
    filtered_survey_scores = {k: v for k, v in survey_scores.items() if k in df['Session Name'].values}

    # Prepare data for plotting
    plot_data = {question: [] for question in ["5_1", "5_2", "6", "7"]}

    for session, scores in filtered_survey_scores.items():
        for question, score in scores.items():
            plot_data[question].append(score)

    # Plot the distributions for each question
    plt.figure(figsize=(10, 6))

    for i, question in enumerate(["5_1", "5_2", "6", "7"], 1):
        plt.subplot(2, 2, i)
        sns.histplot(plot_data[question], bins=10, color='skyblue')
        plt.title(f"Distribution of {question} Scores")
        plt.xlabel(f"{question} Score")
        plt.ylabel("Frequency")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated/alsfrsr_scores.json"  # Replace with actual path
    original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated/output_v2.csv"  # CSV from previous script

    survey_scores = load_survey_scores(survey_json_path)
    plot_survey_score_distribution(survey_scores, original_csv)
