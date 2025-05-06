import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
data_nonabs = pd.read_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/dominant_hand/classification_info_compiled.csv')
data_abs = pd.read_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/classification_results/abs/classification_info_compiled.csv')

# # Calculate mean and standard deviation for Velocity, Acceleration, and Jerk for both cohorts
# nonabs_stats = data_nonabs.groupby('Cohort')[['Velocity', 'Acceleration', 'Jerk']].agg(['mean', 'std']).reset_index()
# nonabs_stats.columns = ['Cohort', 'Velocity_mean', 'Velocity_std', 'Acceleration_mean', 'Acceleration_std', 'Jerk_mean', 'Jerk_std']
#
# abs_stats = data_abs.groupby('Cohort')[['Velocity', 'Acceleration', 'Jerk']].agg(['mean', 'std']).reset_index()
# abs_stats.columns = ['Cohort', 'Velocity_mean', 'Velocity_std', 'Acceleration_mean', 'Acceleration_std', 'Jerk_mean', 'Jerk_std']
#
# # Prepare data for plotting
# metrics = ['Velocity', 'Acceleration', 'Jerk']
# fig, axs = plt.subplots(3, 2, figsize=(12, 18))
#
# for i, metric in enumerate(metrics):
#     # Non-abs data
#     nonabs_patient_data = [nonabs_stats.loc[1, f'{metric}_mean'], nonabs_stats.loc[1, f'{metric}_std']]
#     nonabs_control_data = [nonabs_stats.loc[0, f'{metric}_mean'], nonabs_stats.loc[0, f'{metric}_std']]
#     axs[i, 0].boxplot([nonabs_patient_data, nonabs_control_data], labels=['Patient', 'Control'])
#     axs[i, 0].set_title(f'{metric} (Non-abs)')
#     axs[i, 0].set_ylabel(metric)
#
#     # Abs data
#     abs_patient_data = [abs_stats.loc[1, f'{metric}_mean'], abs_stats.loc[1, f'{metric}_std']]
#     abs_control_data = [abs_stats.loc[0, f'{metric}_mean'], abs_stats.loc[0, f'{metric}_std']]
#     axs[i, 1].boxplot([abs_patient_data, abs_control_data], labels=['Patient', 'Control'])
#     axs[i, 1].set_title(f'{metric} (Abs)')
#     axs[i, 1].set_ylabel(metric)
#
# plt.tight_layout()
# plt.show()


nonabs_stats = data_nonabs.groupby('Cohort')[['Velocity', 'Acceleration', 'Jerk']].agg(['mean', 'std']).reset_index()
nonabs_stats.columns = ['Cohort', 'Velocity_mean', 'Velocity_std', 'Acceleration_mean', 'Acceleration_std', 'Jerk_mean', 'Jerk_std']

abs_stats = data_abs.groupby('Cohort')[['Velocity', 'Acceleration', 'Jerk']].agg(['mean', 'std']).reset_index()
abs_stats.columns = ['Cohort', 'Velocity_mean', 'Velocity_std', 'Acceleration_mean', 'Acceleration_std', 'Jerk_mean', 'Jerk_std']

# Prepare data for plotting
metrics = ['Velocity', 'Acceleration', 'Jerk']
fig, axs = plt.subplots(3, 2, figsize=(12, 18))

for i, metric in enumerate(metrics):
    # Non-abs data
    nonabs_patient_data = [nonabs_stats.loc[1, f'{metric}_mean'], nonabs_stats.loc[1, f'{metric}_std']]
    nonabs_control_data = [nonabs_stats.loc[0, f'{metric}_mean'], nonabs_stats.loc[0, f'{metric}_std']]
    axs[i, 0].boxplot([nonabs_patient_data, nonabs_control_data], labels=['Patient', 'Control'])
    axs[i, 0].set_title(f'{metric} (Non-abs)')
    axs[i, 0].set_ylabel(metric)

    # Abs data
    abs_patient_data = [abs_stats.loc[1, f'{metric}_mean'], abs_stats.loc[1, f'{metric}_std']]
    abs_control_data = [abs_stats.loc[0, f'{metric}_mean'], abs_stats.loc[0, f'{metric}_std']]
    axs[i, 1].boxplot([abs_patient_data, abs_control_data], labels=['Patient', 'Control'])
    axs[i, 1].set_title(f'{metric} (Abs)')
    axs[i, 1].set_ylabel(metric)

    # Set the same y-axis range for both plots in the row
    min_y = min(axs[i, 0].get_ylim()[0], axs[i, 1].get_ylim()[0])
    max_y = max(axs[i, 0].get_ylim()[1], axs[i, 1].get_ylim()[1])
    axs[i, 0].set_ylim(min_y, max_y)
    axs[i, 1].set_ylim(min_y, max_y)

plt.tight_layout()
plt.show()