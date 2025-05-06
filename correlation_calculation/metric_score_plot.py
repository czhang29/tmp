import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load data
metric_df = pd.read_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/mediapipe_pose_results/none/metrics_face.csv')
with open('/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_20250313.json') as f:
    score_data = json.load(f)

cohort_df = pd.read_csv( "/home/czhang/PycharmProjects/ModalityAI/ADL/participants_20250317.csv")
# Convert JSON scores to DataFrame
score_df = pd.DataFrame.from_dict(score_data, orient='index')
score_df.index.name = 'Session_ID'
score_df = score_df.reset_index()

# Convert scores to numeric (handling 'no' responses)
for col in score_df.columns:
    if col != 'Session_ID':
        score_df[col] = pd.to_numeric(score_df[col], errors='coerce')

# Merge metrics with scores
merged_df = pd.merge(metric_df, score_df, on='Session_ID')
merged_df = merged_df.merge(cohort_df, left_on="Session_ID", right_on="Session ID", how="left")
# merged_df = merged_df[merged_df['Percentage Frames Used in Video'] >= 0]
merged_df = merged_df[merged_df['Cohort'] == 'patient']
# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Questions to analyze
questions = ['5_2', '6', '7']

for i, question in enumerate(questions):
    tmp_df = merged_df[['Velocity', question]].dropna()

    # Calculate Spearman correlation
    r = stats.pearsonr(tmp_df['Velocity'], tmp_df[question])[0]

    # Create plot
    sns.scatterplot(data=tmp_df, x='Velocity', y=question, ax=axes[i])
    sns.regplot(data=tmp_df, x='Velocity', y=question,
                scatter=False, ci=95,
                line_kws={'color': 'grey'}, ax=axes[i])

    axes[i].set_title(f'Question {question}\n(n={len(tmp_df)})')
    axes[i].set_xlabel(f'Velocity (Spearman r={r:.2f})')
    axes[i].set_ylabel('ALSFRS-R Score')

plt.tight_layout()
plt.show()
