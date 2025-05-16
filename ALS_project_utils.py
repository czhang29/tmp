import pandas as pd
import numpy as np
import logging

def preprocess_demographics_and_diagnostics_information(dataframe, config):
    """
    Perform project-specific preprocessing after initial loading and merging.
    Input:
        dataframe (pd.DataFrame): DataFrame after loading data and merging demographics.
        config (configparser.ConfigParser): The configuration object.
    Output:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logging.info("Running ALS_project_utils.preprocess_demographics_and_diagnostics_information")
    
    # Add any specific cleaning, calculations, or feature engineering needed
    # for the ALS data *before* the main analysis starts.
    # For example, if you needed to calculate days since onset and had the date:
    # if 'demo_date_of_onset' in dataframe.columns:
    #     dataframe['demo_date_of_onset'] = pd.to_datetime(dataframe['demo_date_of_onset'], errors='coerce')
    #     # Ensure 'session_info_timestamp' is datetime
    #     dataframe['session_info_timestamp'] = pd.to_datetime(dataframe['session_info_timestamp'], errors='coerce')
    #     dataframe['demo_days_since_onset'] = (dataframe['session_info_timestamp'] - dataframe['demo_date_of_onset']).dt.days

    # If no specific preprocessing is needed at this stage beyond what load_and_preprocess_data does,
    # simply return the dataframe.
    return dataframe

def assign_cohort(df_row, config):
    """
    Assigns an analysis cohort index and name based on data in the row.
    Control group should have the *highest* integer index.
    Input:
        df_row (pd.Series): A row of the DataFrame.
        config (configparser.ConfigParser): The configuration object.
    Output:
        tuple: (cohort_index, cohort_name)
    """
    # 'demo_patient_flag' is created by utils.load_and_preprocess_data based on
    # the 'patient_indicator_column' and 'patient_indicator_value' in the config.
    # It should be 1 for patients and 0 for controls.
    
    if 'demo_patient_flag' not in df_row:
        logging.error("Column 'demo_patient_flag' not found in DataFrame row during cohort assignment.")
        return np.nan, 'Error'
        
    if pd.isna(df_row['demo_patient_flag']):
        return np.nan, 'Unknown' # Handle cases where patient flag couldn't be determined

    if int(df_row['demo_patient_flag']) == 0:
        # Assign Control cohort index (highest value, e.g., 1 for 2 cohorts)
        return 1, 'Control'
    elif int(df_row['demo_patient_flag']) == 1:
        # Assign Patient cohort index (lower value, e.g., 0 for 2 cohorts)
        return 0, 'Patient'
    else:
        # Handle unexpected values
        return np.nan, 'Unknown'

# Add any other project-specific utility functions if needed.
# For example, if cohort assignment might change over time based on scores,
# you might need an 'adapt_cohort_assignments' function similar to the
# PeterCohenFoundation example, but this depends on your analysis needs.
# def adapt_cohort_assignments(dataframe, config):
#     # ... implementation based on your logic ...
#     return dataframe

