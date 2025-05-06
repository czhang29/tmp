import json

# Load the JSON data from the provided file
file_name = '/home/czhang/PycharmProjects/ModalityAI/ADL/alsfrsr_scores_crowdsourced_20250320.json'
with open(file_name, 'r') as file:
    data = json.load(file)

# Analyze the JSON data
def analyze_sessions(data):
    total_sessions = len(data)
    sessions_with_score_less_than_4 = 0
    sessions_with_all_scores_4 = 0

    for session_id, scores in data.items():
        # Check if any score is less than 4 in survey questions 5_2, 6, and 7
        relevant_scores = [scores.get('5_2'), scores.get('6'), scores.get('7')]
        if any(score is not None and int(score) < 4 for score in relevant_scores):
            sessions_with_score_less_than_4 += 1
            print(session_id)
        elif '5_2' not in scores and '5_1' in scores and int(scores['5_1']) < 4:
            # Check for 5_1 if 5_2 is missing
            sessions_with_score_less_than_4 += 1
            print(session_id)
        else:
            sessions_with_all_scores_4 += 1

    return total_sessions, sessions_with_score_less_than_4, sessions_with_all_scores_4

# Perform the analysis
total_sessions, sessions_with_score_less_than_4, sessions_with_all_scores_4 = analyze_sessions(data)

# Output the results
print(f"Total Sessions: {total_sessions}")
print(f"Sessions with at least one score < 4 in 5_2, 6, or 7 (or 5_1 if 5_2 is missing): {sessions_with_score_less_than_4}")
print(f"Sessions with all scores >= 4 in 5_2, 6, and 7: {sessions_with_all_scores_4}")