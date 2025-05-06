import pandas as pd
import re

# Load the merged data and clinician rating CSV files
merged_data = pd.read_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/merged_data_20250319.csv')
clinician_rating = pd.read_csv('/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_clinician_rating_20250320.csv')

# Extract relevant columns from merged data
merged_data_relevant = merged_data[
    ['participant_id', 'timestamp', 'alsfrsr_5_1', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']].copy()

# Extract relevant columns from clinician rating
clinician_rating_relevant = clinician_rating.iloc[3:, [1, 2, 11, 12, 13, 14]].copy()

# Rename columns for consistency
clinician_rating_relevant.columns = [
    'visit_date',
    'participant_id',
    'alsfrsr_5_2',
    'alsfrsr_5_1',
    'alsfrsr_6',
    'alsfrsr_7'
]


# Function to extract numerical score from descriptions
def extract_score(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        if match:
            return int(match.group())
    return None


# Apply the function to extract scores for relevant columns in clinician ratings
for column in ['alsfrsr_5_2', 'alsfrsr_5_1', 'alsfrsr_6', 'alsfrsr_7']:
    clinician_rating_relevant[column] = clinician_rating_relevant[column].apply(extract_score)

# Convert timestamps to datetime format and specify the date format explicitly for visit_date
clinician_rating_relevant['visit_date'] = pd.to_datetime(
    clinician_rating_relevant['visit_date'], format='%m/%d/%y', errors='coerce'
)
clinician_rating_relevant.dropna(subset=['visit_date'], inplace=True)

merged_data_relevant['timestamp'] = pd.to_datetime(merged_data_relevant['timestamp'], errors='coerce')

# Find the closest timestamp in merged data for each participant in clinician rating
comparison_results = []
for _, clinician_row in clinician_rating_relevant.iterrows():
    participant_id = clinician_row['participant_id']
    visit_date = clinician_row['visit_date']

    # Filter merged data for the same participant
    participant_sessions = merged_data_relevant[merged_data_relevant['participant_id'] == participant_id].copy()

    if not participant_sessions.empty:
        # Find the session with the closest timestamp
        participant_sessions['time_diff'] = (participant_sessions['timestamp'] - visit_date).abs()
        closest_session = participant_sessions.loc[participant_sessions['time_diff'].idxmin()]

        # Append the comparison results
        comparison_results.append({
            'participant_id': participant_id,
            'visit_date': visit_date,
            'merged_session': closest_session
        })

# Perform the comparison for the four ALSFRS-R questions
comparison_results_final = []
for result in comparison_results:
    participant_id = result['participant_id']
    visit_date = result['visit_date']
    merged_session = result['merged_session']

    comparison = {'participant_id': participant_id}
    for question in ['alsfrsr_5_1', 'alsfrsr_5_2', 'alsfrsr_6', 'alsfrsr_7']:
        clinician_value = clinician_rating_relevant.loc[
            (clinician_rating_relevant['participant_id'] == participant_id) &
            (clinician_rating_relevant['visit_date'] == visit_date),
            question
        ].values

        merged_value = merged_session[question]

        # Handle "no" in merged data as equivalent to NaN in clinician rating file for alsfrsr_5_1
        if question == 'alsfrsr_5_1' and merged_value == "no" and len(clinician_value) > 0 and pd.isna(
                clinician_value[0]):
            comparison[question] = ""  # Leave cell empty if they are equal
        else:
            try:
                if len(clinician_value) > 0 and clinician_value[
                    0] is not None:  # Ensure clinician_value has at least one element and is not None
                    clinician_numeric = float(clinician_value[0])
                    merged_numeric = float(merged_value)
                    comparison[question] = merged_numeric - clinician_numeric
                else:
                    comparison[question] = f"{merged_value}/nan"
            except (ValueError, TypeError):
                comparison[question] = f"{merged_value}/{clinician_value}"

    comparison_results_final.append(comparison)

# Convert the results to a DataFrame and save to CSV file
comparison_df = pd.DataFrame(comparison_results_final)
comparison_df.to_csv(
    '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_clinician_rating_comparison_results_with_extracted_scores.csv',
    index=False)

print(comparison_df.head())
