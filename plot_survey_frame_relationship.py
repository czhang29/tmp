import json
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


def merge_survey_with_csv(original_csv, survey_scores):
    """Merge the original CSV with survey scores and return a DataFrame."""
    df = pd.read_csv(original_csv)

    # Merge survey scores into DataFrame
    df = df.merge(pd.DataFrame.from_dict(survey_scores, orient="index"),
                  left_on="Session Name", right_index=True, how="left")

    # Convert survey scores to numeric (handle "no" as NaN)
    df[["5_1", "5_2", "6", "7"]] = df[["5_1", "5_2", "6", "7"]].apply(pd.to_numeric, errors='coerce')

    return df


def plot_scatter(df, y_variable, y_label, title_suffix):
    """Plot scatter plots for survey scores vs. a given metric."""
    survey_questions = ["5_1", "5_2", "6", "7"]

    plt.figure(figsize=(12, 8))

    for i, question in enumerate(survey_questions, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(data=df, x=question, y=y_variable, alpha=0.6, edgecolor=None)
        plt.xlabel(f"{question} Score")
        plt.ylabel(y_label)
        plt.title(f"{question} Score vs. {title_suffix}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    survey_json_path = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/alsfrsr_scores.json"  # Replace with actual path
    original_csv = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/output_v2.csv"  # CSV from previous script

    survey_scores = load_survey_scores(survey_json_path)
    df = merge_survey_with_csv(original_csv, survey_scores)

    # Scatter plot: Survey Score vs. Number of Frames Used
    plot_scatter(df, y_variable="Number of Frames Used", y_label="Number of Frames Used",
                 title_suffix="Number of Frames Used")

    # Scatter plot: Survey Score vs. Percentage of Frames Used
    plot_scatter(df, y_variable="Percentage Frames Used", y_label="Percentage of Frames Used",
                 title_suffix="Percentage of Frames Used")
