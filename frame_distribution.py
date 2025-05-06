import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_frames_distribution(csv_file, output_dir):
    # Load data
    df = pd.read_csv(csv_file)

    # Check if the necessary column exists
    if "Percentage Frames Used in Video" not in df.columns:
        raise ValueError("The column 'Number of Frames Used' is missing from the CSV file.")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Percentage Frames Used in Video"]*100, bins=30)
    plt.xlabel("Percentage of Frames Used")
    plt.ylabel("Frequency")
    plt.title("Distribution of Percentage Frames Used")
    plt.savefig(f"{output_dir}/percentage_frames_distribution_histogram.png")
    plt.show()

    # Plot box plot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=(df["Percentage Frames Used in Video"]*100))
    plt.xlabel("Percentage of Frames Used")
    plt.title("Boxplot of Percentage Frames Used")
    plt.savefig(f"{output_dir}/percentage_frames_distribution_boxplot.png")
    plt.show()


if __name__ == "__main__":
    csv_file = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe/output_v2.csv"  # Update path
    output_dir = "/home/czhang/PycharmProjects/ModalityAI/ADL/annotated_Mediapipe"  # Update path
    plot_frames_distribution(csv_file, output_dir)
