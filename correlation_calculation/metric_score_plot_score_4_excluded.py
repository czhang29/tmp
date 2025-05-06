import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
data_path = '/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv'
df = pd.read_csv(data_path)

# Filter sessions where mediapipe_hand_face_percentage_frames_used_in_video >= 0.1
filtered_df = df

# Define velocity metrics and ALSFRS-R questions
velocity_metrics = ['mediapipe_pose_no_filter_face_velocity']
questions = ['alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']

# Create plots for each velocity metric
for velocity in velocity_metrics:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, question in enumerate(questions):
        # Filter valid data for plotting
        # tmp_df = filtered_df[[velocity, question]].dropna()
        tmp_df = filtered_df[filtered_df[question] < 4][[velocity, question]].dropna()

        # Calculate Pearson correlation
        r = stats.spearmanr(tmp_df[velocity], tmp_df[question])[0]

        # Create scatter plot and regression line
        sns.scatterplot(data=tmp_df, x=velocity, y=question, ax=axes[i])
        sns.regplot(data=tmp_df, x=velocity, y=question,
                    scatter=False, ci=95,
                    line_kws={'color': 'grey'}, ax=axes[i])

        axes[i].set_title(f'Question {question}\n(n={len(tmp_df)})')
        axes[i].set_xlabel(f'{velocity} (Spearman r={r:.2f})')
        axes[i].set_ylabel('ALSFRS-R Score')
        axes[i].set_xlim(0,100)

    plt.tight_layout()
    plt.show()
