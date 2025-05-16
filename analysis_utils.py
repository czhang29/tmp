"""This module contains functions commonly used in Modality's data analyses."""

import ast
from copy import deepcopy
from collections import OrderedDict
import datetime
import json
import logging
from math import nan
from readline import insert_text
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pprint
import re
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
import scipy.stats as stats
import seaborn as sns
from sklearn.utils import arrayfuncs
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn import metrics


class SexSpecificScaler():
    """Normalization like sklearn's StandardScaler, but separately for male and female speakers.
    
    Requires the column "demo_sex" in the dataframe, which must contain values {'Male', 'Female'}
    """
    def _reset(self):
        if hasattr(self, "mean_m_"):
            del self.mean_m_
            del self.mean_f_
            del self.var_m_
            del self.var_f_

    def fit(self, X, y=None):
        # Reset internal state before fitting
        self._reset()

        X_m = X[X['demo_sex']=='Male'].drop(columns=['demo_sex'])
        X_f = X[X['demo_sex']=='Female'].drop(columns=['demo_sex'])
        self.mean_m_ = X_m.mean(axis=0, skipna=True)
        self.mean_f_ = X_f.mean(axis=0, skipna=True)
        self.var_m_ = X_m.std(axis=0, skipna=True) + 0.000000000001
        self.var_f_ = X_f.std(axis=0, skipna=True) + 0.000000000001

        return self

    def transform(self, X):
        X_m = X[X['demo_sex']=='Male'].drop(columns=['demo_sex'])
        X_f = X[X['demo_sex']=='Female'].drop(columns=['demo_sex'])
        if not X_m.empty:
            X_m = (X_m - self.mean_m_) / self.var_m_
        if not X_f.empty:
            X_f = (X_f - self.mean_f_) / self.var_f_
        return pd.concat([X_m, X_f]).reindex(X.index)


def select_controls(dataframe, patient_user_ids, sex_matched=True, age_matched=True, age_tolerance=0, min_study_duration=0):
    """Given a dataframe and a set of patient user IDs, find age and/or sex matched controls.

    Parameters:
        dataframe (pandas DataFrame): The complete dataframe to be processed (column 'demo_patient_flag' is required!)
        patient_user_ids (list): list of user IDs for which sex/age matched controls should be selected
        sex_matched (bool): if True, look for a set of sex matched controls (default: True)
        age_matched (bool): if True, look for a set of age matched controls (default: True)
        age_tolerance (int): how many years deviation are OK if exact match is not found (default: 0)
        min_study_duration (int): optional; how many days should at least be between first and last session

    Returns:
        selected_control_ids (list): selected control IDs

    If both sex_matched and age_matched are False, controls are selected at random so that the number of control IDs matched the number of patient IDs.
    """
    def search_match(dataframe, available_control_ids, sex_matched, sex, age_matched, age, age_tolerance, min_study_duration):
        for control_id in available_control_ids:
            control_data = dataframe[dataframe['session_info_access_code']==control_id].sort_values(['session_info_timestamp'])
            study_duration = (control_data.iloc[-1]['session_info_timestamp'] - control_data.iloc[0]['session_info_timestamp']).days
            # some users don't have age info - for those, just take a sex-matched control
            if (not sex_matched or control_data['demo_sex'].iloc[0] == sex) and np.isnan(age):
                if study_duration >= min_study_duration:
                    return control_id
            if (not sex_matched or control_data['demo_sex'].iloc[0] == sex) \
                and (not age_matched or control_data['demo_age'].iloc[0] in np.arange(age - age_tolerance, age + age_tolerance + 1)):
                if study_duration >= min_study_duration:
                    return control_id
        return None

    #available_control_ids = list(dataframe[dataframe['demo_patient_flag']==0]['session_info_access_code'].unique())
    # do value_count(sort=True) first, so that control users with more sessions are preferred
    available_control_ids = list(dataframe[dataframe['demo_patient_flag']==0]['session_info_access_code'].value_counts().index)
    selected_control_ids = []
    if len(set(patient_user_ids)) > len(available_control_ids):
        logging.warning("WARNING: less controls than patients are available.")
    if sex_matched and not 'demo_sex' in dataframe.columns.to_list():
        logging.error("Aborted. dataframe does not contain column 'demo_sex' but sex_matched is True")
        return dataframe
    if age_matched and not 'demo_age' in dataframe.columns.to_list():
        logging.error("Aborted. dataframe does not contain column 'demo_age' but age_matched is True")
        return dataframe

    for patient_id in patient_user_ids:
        user_data = dataframe[dataframe['session_info_access_code']==patient_id].sort_values(['session_info_timestamp'])
        if sex_matched:
            sex = user_data['demo_sex'].iloc[0]
        else:
            sex = np.nan
        if age_matched:
            age = user_data['demo_age'].iloc[0]
        else:
            age = np.nan
        found_match = False
        #for control_id in available_control_ids:
        #    control_data = dataframe[dataframe['session_info_access_code']==control_id].sort_values(['session_info_timestamp'])
        #    # some users don't have age info - for those, just take a sex-matched control
        #    if (not sex_matched or control_data['demo_sex'].iloc[0] == sex) \
        #        and (not age_matched or control_data['demo_age'].iloc[0] == age or np.isnan(age)):
        #        selected_control_ids.append(control_id)
        #        available_control_ids.remove(control_id)
        #        found_match = True
        #    if found_match:
        #        break
        #if not found_match and age_tolerance > 0 and age_matched:
        #    logging.info(f"Did not find exact age match for user {patient_id}, looking for a control in the range [age - {age_tolerance} years, age + {age_tolerance} years]") 
        for tolerance in range(0, age_tolerance+1):
            control_id = search_match(dataframe, available_control_ids, sex_matched, sex, age_matched, age, tolerance, min_study_duration)
            if control_id is not None:
                found_match = True
                selected_control_ids.append(control_id)
                available_control_ids.remove(control_id)
                break

            #for control_id in available_control_ids:
            #    control_data = dataframe[dataframe['session_info_access_code']==control_id].sort_values(['session_info_timestamp'])
            #    # some users don't have age info - for those, just take a sex-matched control
            #    if (not sex_matched or control_data['demo_sex'].iloc[0] == sex) \
            #        and (not age_matched or control_data['demo_age'].iloc[0] in np.arange(age - age_tolerance, age + age_tolerance + 1) or np.isnan(age)):
            #        selected_control_ids.append(control_id)
            #        available_control_ids.remove(control_id)
            #        found_match = True
            #    if found_match:
            #        break
        if not found_match:
            logging.info(f"Did not find a match for user {patient_id} ({sex}, {age} years)")

    logging.info(f"Found {len(selected_control_ids)} controls (for {len(patient_user_ids)} patients)")

    return selected_control_ids


def estimate_slope_and_intercept(dataframe, x_axis: str, y_axis: str, anchor_value=None):
    """Compute linear regression slope and intercept.

    Parameters:
        dataframe (pandas DataFrame): Contains the observations as rows
        x_axis (str): Column name of dataframe to use as X-axis
        y_axis (str): Column name of dataframe to use as Y-axis (target)
        anchor_value (float): additional y-value at x=0, can be used as 'onset anchor', e.g. for survey scores at time of disease onset

    Returns:
        slope (float): linear regression slope
        intercept (float): intercept term

    If there are less than 3 observations, (None, None) is returned.
    """
    X = dataframe[x_axis]
    Y = dataframe[y_axis]
    if anchor_value:
        X = np.append(X, [0])
        Y = np.append(Y, [anchor_value])
    idx = np.isfinite(X) & np.isfinite(Y)
    if idx.any() and sum(idx) >= 3:
        slope, intercept = np.polyfit(X[idx], Y[idx], 1)
        return slope, intercept
    else:
        return None, None



def compute_correlation_between_slopes(user_ids, variables: list, dataframes, x_axis, minimum_num_samples=3, minimum_duration=0):
    # user_ids: list of ids
    # variables: list of column names (vars that should be compared)
    # dataframes: list of DFs (same len as variables), was designed so that data could be in different dataframes -> should only be one df
    # minimum_duration: right now, this assumes that x_axis is of type datetime, should be more general

    # returns: spearman rho (probably good to specify as an argument which corrlation should be computed)
    # p: p-value
    # all_slopes: dataframe with len(variables) columns containing all slopes (num rows should be num user IDs)
    all_slopes = pd.DataFrame(columns=variables)
    for user_id in user_ids:
        data = [df[df['session_info_access_code']==user_id] for df in dataframes]
        if all(len(x) >= minimum_num_samples for x in data) and \
            all((x[x_axis].max() - x[x_axis].min()) >= minimum_duration for x in data):
            slopes_for_user = []
            for var, df in zip(variables, data):
                slope, intercept = estimate_slope_and_intercept(
                    dataframe=df,
                    x_axis=x_axis,
                    y_axis=var)
                slopes_for_user.append(slope)
            if len(slopes_for_user) >= 2:
                #all_slopes = all_slopes.append(pd.Series(slopes_for_user, index = all_slopes.columns), ignore_index=True)
                all_slopes.loc[user_id] = pd.Series(slopes_for_user, index = all_slopes.columns)
    r = all_slopes[variables[0]].corr(all_slopes[variables[1]])
    rho, p = stats.spearmanr(all_slopes)
    #r_lag_1 = all_slopes[variables[0]].corr(all_slopes[variables[1]].shift(1))  # attempt to do cross-correlation, but this is does not work because of the varying time spans between two measurements across participants!
    #if np.abs(r_lag_1) > 0.3:
    #    print(f"{metric}: r(lag=1) = {r_lag_1} (n={len(all_slopes)})")
    return rho, p, all_slopes


def aggregate_metrics(dataframe, metrics, prefix=''):
    """For given metrics, aggregate the values over all tasks by taking the mean.

    Parameters:
        dataframe (pandas DataFrame): Contains the data, with different metrics as columns
        metrics (list): List of all the metrics that shall be aggregated
        prefix (str): prefix of the metrics category to prepend to all metric labels, e.g. 'speech_metrics'

    Returns:
        dataframe (pandas DataFrame): new dataframe with aggregated metrics. All other columns remain unchanged

    The mean aggregation works by filtering columns based on the metric label.
    If a metric name is entailed in another one's name, this can cause problems!
    """
    for m in metrics:
        if isinstance(m, float):
            continue
        # remove duration metrics from the list (e.g. speaking_time, articulation_time), because they should never be aggregated across tasks
        if 'time' in m:
            metrics.remove(m)
    for m in metrics:
        if isinstance(m, float):
            continue
        # special treatment because 'eye_blinks' is also contained in 'eye_blinks_right/left' -> use negative lookahead in regex
        # same with 'open_max' and 'eye_open_max' -> use negative lookbehind in regex
        if m == 'eye_blinks':
            regex = f'.*{m}(?!_left|_right).*'
        elif m.startswith('open'):
            # RND-2395 - a negative lookBEHIND is needed in the following, instead of a lookahead (see https://www.rexegg.com/regex-lookarounds.html )
            regex = f'.*(?<!eye)_{m}.*'
        else:
            regex = f'.*{m}.*'
        if not dataframe.filter(regex=regex).empty:
            new = pd.DataFrame(np.mean(dataframe.filter(regex=regex), axis=1))
            new.columns = [f'{prefix if prefix[-1] != "_" else prefix[:-1]}_{m}']
            dataframe = dataframe.drop(columns=dataframe.filter(regex=regex).columns)
            dataframe = pd.concat((dataframe, new), axis=1)
    return dataframe


def zscore_by_sex(df, metrics_prefixes, sex_column_label):
    """Z-score normalization on a dataframe separately for male and female users.

    Parameters:
        df (pandas DataFrame): Contains the data
        metrics_prefixes (list[str]): metric category prefixes to filter for, e.g. 'speech_metrics', 'facial_metrics'
        sex_column_label (str): name of column in dataframe that contains sex information

    Returns:
        df (pandas DataFrame): new dataframe with z-scored metrics. All other columns remain unchanged

    Specifying metrics_prefixes allows to normalize one or more metrics types. E.g., one could z-score only speech metrics, while leaving facial metrics unchanged.
    The column 'sex_column_label' can contain any date type, e.g. str ("female"/"male") or int (0/1), as long as it can be used by the pandas groupby() function to form separate groups.
    """
    excluded_columns = df.filter(regex='^(?!' + '|'.join(metrics_prefixes) + ').*').columns.to_list()
    excluded_part = df[excluded_columns]
    excluded_columns.remove(sex_column_label)  #remove this from the list before dropping columns
    df = df.drop(columns=excluded_columns)
    transformed = df.groupby(sex_column_label).transform(
        lambda x: (x - x.mean(skipna=True)) / (x.std(skipna=True) + 0.000000001))
    df = transformed.merge(excluded_part, right_index=True, left_index=True)
    return df


def windowed_average_slope(dataframe, time_col, metric, window_size=30):
    """Not yet implemented.

    The idea for this was to create a function that outputs an average slope over a sliding window through time, in order to smoothen the slopes.
    """
    # pandas' function DataFrame.rolling might be useful, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    dataframe = dataframe.sort_values('time_col')
    if (dataframe.iloc[-1][time_col] - dataframe.iloc[0][time_col]) <= window_size:
        pass


def pearsonr_with_pvalues(df):
    """Compute pairwise Pearson correlation of columns, excluding NA/null values.

    Parameters:
        df (pandas DataFrame): contains the observations (column-wise)

    Returns:
        correlation matrix (pandas DataFrame),
        p-value matrix (pandas DataFrame)

    This is an extension to the pandas function DataFrame.corr(), which does not return p-values.
    Note, it could be extended to compute different kinds of correlations (spearman, kendall, ...) based on a keyword argument.
    """
    corr = df.corr()
    # the following function is from https://stackoverflow.com/a/45507587
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
    return corr, pvalues


def plot_effect_sizes(dataframe, filepath, name_suffix='', figsize=(12,8), xlabel=r"Effect size (Glass' $\Delta$)", font_colors=None):
    """Function to plot effect sizes with confidence intervals.

    Parameters:
        dataframe (pandas DataFrame): must contain these columns: effect_size (float), metric (str), error (float), label (str)
            (label is used for the "hue" parameter in seaborn.pointplot, and for the legend; it's usually the cohort pair, e.g. "Controls - pALS")
        filepath (str): location where to store the png file (without actual file name)
        name_suffix (str): optional suffix for the file name; file name will be "effect_sizes_{name_suffix}.png"
        figsize (tuple): figure size in inches
        xlabel (str): label for x-axis
        font_colors (dict[str: matplorlib.color]): optional, dictionary that maps metric prefixes to font colors, e.g. {'speech_metrics': 'brown'}
            (if font_colors is provided, the labels will be colored and the metric type prefixes will be stripped off)

    Returns:
        No return values. The figure plot is shown and saved to disk.
    """
    fig = plt.figure()
    #plt.rcParams.update({'font.size': 14})
    ax = sns.pointplot(x='effect_size', y='metric', hue='label', data=dataframe, join=False, dodge=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Metrics')
    if len(dataframe['label'].unique()) > 1:
        ax.legend(title='Subject groups', fontsize=9, loc='best')
    else:
        ax.get_legend().remove()
    plt.axvline(0, 0, 1, linestyle='--', alpha=0.6)
    plt.axvline(0.5, 0, 1, linestyle='--', alpha=0.3, color='gray')
    plt.axvline(-0.5, 0, 1, linestyle='--', alpha=0.3, color='gray')
    plt.grid(alpha=0.3, linestyle='--', lw=.5)
    if 'error' in dataframe.columns:
        # taken from https://stackoverflow.com/questions/43159528/error-bars-with-seaborn-and-stripplot
        x_coords = []
        y_coords = []
        colors = []
        for point_pair in ax.collections:
            for x, y in point_pair.get_offsets():
                colors.append(point_pair.get_facecolor())
                x_coords.append(x)
                y_coords.append(y)
        x_coords = np.array(x_coords, dtype='float')
        y_coords = np.array(y_coords, dtype='float')
        ecolors = np.reshape(np.array(colors)[~np.isnan(x_coords)], (-1, 4))
        ax.errorbar(x_coords[~np.isnan(x_coords)], y_coords[~np.isnan(y_coords)],
                    xerr=dataframe['error'],
                    zorder=-1, ls='none', ecolor=ecolors)
    # ytick_label colors:
    # taken from https://stackoverflow.com/questions/24617429/matplotlib-different-colors-for-each-axis-label
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    label_colors = [item.get_color() for item in ax.get_yticklabels()]
    if font_colors:
        for i, text in enumerate(ylabels):
            for prefix, color in font_colors.items():
                if text.startswith(prefix):
                    label_colors[i] = color
        # remove the prefixes
        ylabels = [re.sub("|".join(font_colors.keys()), "", x) for x in ylabels]
        ax.set_yticklabels(ylabels)
    for ytick, color in zip(ax.get_yticklabels(), label_colors):
        ytick.set_color(color)

    plt.xticks(np.arange(np.floor(dataframe['effect_size'].min()), np.ceil(dataframe['effect_size'].max())))
    if plt.gca().get_legend():
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='14')
    fig = plt.gcf()
    fig.set_dpi(72)
    fig.set_size_inches(figsize[0], figsize[1])
    plt.tight_layout()
    plt.savefig(f'{filepath}/effect_sizes{name_suffix}.png', dpi=200, facecolor='white')
    plt.show()
    plt.rcParams.update({'font.size': 11})


def summarize_data(df):
    """Print a summary of the dataframe.

    Parameters:
        df (pandas DataFrame): contains the data to be summarized

    Returns:
        No return values. Summary is printed out directly.

    Note, right now the function assumes certain column names; needs to be adapted (it was taken from my APST notebook).
    """
    counts = {
        'all_sessions': None,
        'unique_users': None
    }
    counts['all_sessions'] = len(df)
    counts['unique_users'] = len(df['session_info_access_code'].unique())
    first_session_from_each_user = df.sort_values(['session_info_access_code', 'session_info_timestamp']).drop_duplicates(
        subset='session_info_access_code', keep='first')
    if 'demo_sex' in df.columns.to_list():
        sex_labels = df['demo_sex'].unique()
        for label in sex_labels:
            counts[f'sex-{label}'] = len(first_session_from_each_user[first_session_from_each_user['demo_sex']==label])
    if 'demo_patient_flag' in df.columns.to_list():
        counts['patients'] = sum(first_session_from_each_user['demo_patient_flag']==1)
        counts['controls'] = sum(first_session_from_each_user['demo_patient_flag']==0)
    if 'demo_cohort_name' in df.columns.to_list():
        for cohort in df['demo_cohort_name'].unique():
            counts['cohort: ' + str(cohort)] = sum(first_session_from_each_user['demo_cohort_name']==cohort)
    logging.info("DATA SUMMARY")
    logging.info(pprint.pformat(counts))
    logging.info('\nNumber of sessions per participant:')
    sess_per_user = df.reset_index().groupby('session_info_access_code').count()['session_info_session_id'].sort_values(ascending=False)
    sess_per_user.hist(bins=range(sess_per_user.max() + 1), color='#CC6633')
    plt.xlabel('Number of sessions')
    plt.ylabel('Number of participants')
    logging.info(f"{np.round(sess_per_user.mean(), 2)} +/- {np.round(sess_per_user.std(), 2)} sessions on average per participant (median: {sess_per_user.median()}).")
    plt.show()
    if 'demo_age' in df.columns.to_list():
        logging.info('\nAge distribution:')
        logging.info(first_session_from_each_user['demo_age'].describe())
        first_session_from_each_user['demo_age'].hist(bins=range(0, 100, 10), color='#CC6633')
        plt.xlabel('Age at first session [years]')
        plt.ylabel('Number of participants')
        plt.xticks(range(0, 100, 10))
        plt.show()


def assign_cohort(df):
    """note, this function needs to be implemented by each project, if needed"""
    if df['demo_patient_type'] == 'NALSTYP':
        return 2, 'Controls'
    elif df['survey_response_scores_bulbar'] == 12:
        return 1, 'Bulbar pre-symptomatic'
    elif df['survey_response_scores_bulbar'] < 12:
        return 0, 'Bulbar symptomatic'


def compute_age(birthday, timestamp):
    return timestamp.year - birthday.year - ((timestamp.month, timestamp.day) < (birthday.month, birthday.day))


def reduce_longitudinal_data(dataframe, timestamp_col: str, user_col: str, num_days_to_include: int, ascending=True):
    """Reduces data to the given number of days for each participant.

    Parameters:
        dataframe (pandas DataFrame): data, must contain timestamp_col and user_col (see below)
        timestamp_col (str): name of column in dataframe that contains timestamps
        user_col (str): name of column in dataframe that contains user identifiers
        num_days_to_include (int): number of days to include for each participant (the first num_of_days_to_include days after sorting the dataframe are considered)
        ascending (bool): sort ascending or descending (controls whether the first or last sessions for each user are returned)

    Returns:
        Reduced dataframe. Columns are identical to input, only rows are dropped.
    """
    dataframe = dataframe.sort_values([timestamp_col], ascending=ascending)
    result = pd.DataFrame()
    for user_id in dataframe[user_col].unique():
        user_data = dataframe[dataframe[user_col]==user_id]
        first_timestamp = user_data.iloc[0][timestamp_col]
        if ascending:
            user_data = user_data[user_data[timestamp_col] <= (first_timestamp + datetime.timedelta(days=num_days_to_include))]
        else:
            user_data = user_data[user_data[timestamp_col] >= (first_timestamp - datetime.timedelta(days=num_days_to_include))]
        result = result.append(user_data)
    return result


def get_longitudinal_subset_with_minimum_amount_of_time(dataframe, timestamp_col='session_info_timestamp', user_col='session_info_access_code', days_since_baseline_col='session_info_days_since_baseline', minimum_num_days=90, take_all_user_data=True):
    """Return subset of the dataframe including only users with data worth at least a certain amount of time.

    Parameters:
        dataframe (pandas DataFrame): data, must contain timestamp_col, user_col, and days_since_baseline_col (see below)
        timestamp_col (str): name of column in dataframe that contains timestamps
        user_col (str): name of column in dataframe that contains user identifiers
        days_since_baseline_col (str): name of column in dataframe that contains the number of days since a user's baseline session
        minimum_num_days (int): only users are included that have at least that many days between their first and last session; default: 90
        take_all_user_data (bool): if True, return all data for every user that is included; if False, return only a subset that is truncated after the next session following minimum_num_days for each user.

    Returns:
        Reduced dataframe. Columns are identical to input, only rows are dropped.
    """
    participants = dataframe.sort_values(timestamp_col).groupby(user_col)
    max_days_since_baseline = participants[days_since_baseline_col].aggregate('max')
    users_to_include = max_days_since_baseline[max_days_since_baseline >= minimum_num_days].index
    if take_all_user_data:
        return dataframe[dataframe[user_col].isin(users_to_include)]
    else:
        data_subset = pd.DataFrame()
        for user_id in users_to_include:
            user_data = dataframe[dataframe[user_col]==user_id].sort_values(timestamp_col)
            days_difference = user_data[days_since_baseline_col] - minimum_num_days
            # get the minimum positive value of days_difference, i.e. the closest session in time *after* the given time period
            # then take all data from the user up to this point
            days_actual_interval = days_difference[days_difference >= 0].sort_values().iloc[0] + minimum_num_days
            data_subset = data_subset.append(user_data[user_data[days_since_baseline_col]<=days_actual_interval])
        return data_subset


def plot_scatter_and_residual_plots(x, y, title=''):
    """Basic plotting utility to check for linearity.

    Parameters:
        x (np.array or pandas Series): predictor variable (x-axis)
        y (np.array or pandas Series): target (y-axis), same length as x
        title (str): plot title (default: '')

    No return values.
    """
    plt.scatter(x, y)
    plt.title(title)
    plt.show()
    sns.residplot(x=x,
                  y=y,
                  lowess=True)
    plt.show()


# set up a plotting function to compare variables:
def plot_individual_metrics(data, x_axis: str, variables: list, labels: list, ylims: list, image_file='img.png'):
    if len(variables) not in [2, 3]:
        print('Provide two or three variables to compare to each other')
        return None, None
    if len(variables) != len(labels) or len(variables) != len(ylims):
        print('arguments, "variables", "labels", and "ylims" have to be of the same length')
        return None, None
    if type(data) == list and len(data) != len(variables):
        print('if argument "data" is a list, it must have the same length as "variables"')
        return None, None

    intercepts = []
    slopes = []
    colors = ['blue', 'red', 'orange']
    axes = []
    if type(data) != list:
        data = len(variables) * [data]
    for i in range(len(variables)):
        m, b = estimate_slope_and_intercept(data[i], x_axis, variables[i])
        if m is not None:
            intercepts.append(b)
            slopes.append(m)
        else:
            intercepts.append(0)
            slopes.append(0)
        #print(f"y = {m} * x + {b}")

        # plot the actual variable values as scatter plot
        if i == 0:
            axes.append(data[i].plot(x=x_axis,
                                     y=variables[i],
                                     color=colors[i],
                                     kind='scatter',
                                     legend=False)
                       )
        else:
            axes.append(axes[0].twinx())
            data[i].plot(x=x_axis,
                         y=variables[i],
                         ax=axes[i],
                         color=colors[i],
                         kind='scatter',
                         legend=False)

        # then, plot the fitted linear function in the same color
        axes[i].plot(data[i][x_axis],
                     slopes[i] * data[i][x_axis] + intercepts[i],
                     color=colors[i],
                     linestyle='--',
                     label=labels[i])
        axes[i].set_ylim(ylims[i])
        axes[i].tick_params(axis='y', colors=colors[i])
        if i==2:
            # if there's a third variable, make the y-ticks longer, so they don't overlap with the second
            axes[i].tick_params(length=40)
    axes[0].figure.set_dpi(100)
    axes[0].figure.legend() #loc='upper right', bbox_to_anchor=(.88, .87))
    #plt.xlim(0, 3000)
    fig = plt.gcf()
    fig.set_dpi(72)
    #fig.set_size_inches(30, 6)
    plt.tight_layout()
    plt.savefig(image_file, dpi=200)
    plt.show()
    return slopes, intercepts


def plot_variables_over_time(data, name, variable, x_axis, log_scale=True, hue='session_info_access_code', style=None, y_step=2, x_step=14, y_label=None, filepath=None):
    g = sns.relplot(x=x_axis, y=variable, hue=hue, data=data, kind='line', style=style)
    if log_scale:
        g.set(xscale='log')
        plt.xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    else:
        plt.xticks(range(int(np.floor(data[x_axis].min())), int(np.ceil(data[x_axis].max())) + x_step, x_step), rotation=45)
    if np.ceil(data[variable].max()) == 1 and np.floor(data[variable].min()) == -1:
        # for the case that y values are normalized to be in range [-1; 1]
        plt.yticks(np.arange(-1, 1.1, .1))
    else:
        plt.yticks(range(int(np.floor(data[variable].min())) - 1, int(np.ceil(data[variable].max()+1)) + y_step, y_step))
        plt.ylim(int(np.floor(data[variable].min())) - 1, int(np.ceil(data[variable].max()+1)) + 1)
    plt.grid(linewidth=.3)
    plt.xlabel(x_axis)
    if not y_label:
        plt.ylabel(name)
    else:
        plt.ylabel(y_label)
    plt.title(name)
    plt.axhline(y = 0.0, color = 'gray', linestyle = '--', linewidth=1.0)
    fig = plt.gcf()
    fig.set_dpi(200)
    fig.set_size_inches(25, 8)
    if filepath:
        #plt.tight_layout()
        plt.savefig(filepath, dpi=200)
    plt.show()


def replace_multiple_strings(string, replacement_dict):
    if replacement_dict is not None:
        for key, replacement in replacement_dict.items():
            string = string.replace(key, replacement)
    return string


def k_fold_classification(X, targets, groups, num_folds, class_for_roc=0, return_fold_indices=False, provided_train_idx=None, provided_test_idx=None, plot_filename='ROC_curve.png', num_feats=10, verbose=False):
    n_classes = len(np.unique(targets))
    skf = StratifiedGroupKFold(n_splits=num_folds)
    print(f'---------------------------------------------------')
    test_acc = []
    test_uar = []
    conf_matrices = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    train_indices = {}
    test_indicies = {}
    selected_features = []

    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 11})
    for cv, (train_index, test_index) in enumerate(skf.split(X, targets, groups)):
        print(f'\nCV fold {cv}:')
        # change indices to the provided ones if applicable; probably a really bad workaround
        if provided_train_idx is not None and provided_test_idx is not None:
            train_index = provided_train_idx[cv]
            test_index = provided_test_idx[cv]
        train_indices[cv] = train_index
        test_indicies[cv] = test_index
        # zscore by sex group based on train fold
        train_df = X.iloc[train_index]
        train_male = train_df[train_df['demo_sex']=='Male'].drop(columns=['demo_sex'])
        train_female = train_df[train_df['demo_sex']=='Female'].drop(columns=['demo_sex'])
        test_df = X.iloc[test_index]
        test_male = test_df[test_df['demo_sex']=='Male'].drop(columns=['demo_sex'])
        test_female = test_df[test_df['demo_sex']=='Female'].drop(columns=['demo_sex'])
        m_scaler = StandardScaler()
        if not train_male.empty:
            train_male = pd.DataFrame(m_scaler.fit_transform(train_male), columns=train_male.columns, index=train_male.index)
        if not test_male.empty:
            if train_male.empty:
                test_male = pd.DataFrame(m_scaler.fit_transform(test_male), columns=test_male.columns, index=test_male.index)
            else:
                test_male = pd.DataFrame(m_scaler.transform(test_male), columns=test_male.columns, index=test_male.index)
        f_scaler = StandardScaler()
        if not train_female.empty:
            train_female = pd.DataFrame(f_scaler.fit_transform(train_female), columns=train_female.columns, index=train_female.index)
        if not test_female.empty:
            if train_female.empty:
                test_female = pd.DataFrame(f_scaler.fit_transform(test_female), columns=test_female.columns, index=test_female.index)
            else:
                test_female = pd.DataFrame(f_scaler.transform(test_female), columns=test_female.columns, index=test_female.index)

        X_train = pd.concat([train_male, train_female]).reindex(train_df.index)
        X_test = pd.concat([test_male, test_female]).reindex(test_df.index)
        feature_names = X_train.columns

        if verbose:
            print(f'\nCross validation logistic regression with\n{len(targets)} samples,\n{X_train.shape[1]} features, and\n{num_folds} stratified CV folds.')
            print(f'Class distribution: {np.unique(targets, return_counts=True)}')
        y_train, y_test = targets[train_index], targets[test_index]

        #for col in X_train:
            #print(type(X_train[col]))
       #     for value in X_train[col]:
       #         print(value)
       #         print(type(value))
        #clf = LogisticRegression(max_iter=200) #' random_state=0,
        clf = RandomForestClassifier(random_state=12)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(49))
        #print(clf)
        #clf = svm.SVC(kernel='linear', probability=True)
        #clf = MLPClassifier(random_state=0, max_iter=300)
        # feature selection:
        #clf = RFE(clf, n_features_to_select=num_feats).fit(X_train, y_train)
        for col in X_train:
            for value in X_train[col]:
                assert (type(value) != str), f"col: {col}, value: {value}"
        clf.fit(X_train, y_train)
        #if verbose:
        #    print(f'Selected features:\n{feature_names[clf.support_]}')
        #selected_features.append(feature_names[clf.support_])

        y_score = clf.predict_proba(X_test)
        if n_classes > 2:
            roc_auc_score = metrics.roc_auc_score(y_test, y_score, multi_class='ovo')
            print(f'Overall ROC_AUC_score: {roc_auc_score}')
        # Compute ROC curve and ROC area for each class
#        fpr = dict()
#        tpr = dict()
#        roc_auc = dict()
#        for i in range(n_classes):
#            if n_classes == 2:
#                fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=np.unique(targets)), y_score[:, i])
#            else:
#                fpr[i], tpr[i], _ = metrics.roc_curve(label_binarize(y_test, classes=np.unique(targets))[:, i], y_score[:, i])
#                print(label_binarize(y_test, classes=np.unique(targets)), y_score)
#            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        #idx_group_pair = np.sum(label_binarize(y_test, classes=np.unique(targets))[:, classes_for_roc], axis=1)==1  # take indices for all samples that are in the two desired classes
        #fpr, tpr, _ = metrics.roc_curve(label_binarize(y_test, classes=np.unique(targets))[idx_group_pair, classes_for_roc[0]], y_score[idx_group_pair, classes_for_roc[0]])
        if n_classes == 2:
            fpr, tpr, _ = metrics.roc_curve(label_binarize(y_test, classes=np.unique(targets)), y_score[:, class_for_roc])
        else:
            fpr, tpr, _ = metrics.roc_curve(label_binarize(y_test, classes=np.unique(targets))[:, class_for_roc], y_score[:, class_for_roc])
        roc_auc = metrics.auc(fpr, tpr)
        print(f'ROC_AUC_scores for individual classes: {roc_auc}')
        interp_tpr = np.interp(mean_fpr, fpr, tpr) #fpr[class_for_roc], tpr[class_for_roc])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc) #[class_for_roc])
        ax.plot(fpr, tpr, alpha=0.4, lw=1, #fpr[class_for_roc], tpr[class_for_roc], alpha=.4, lw=1,
                label=f'ROC fold {cv+1} (AUC = {np.round(roc_auc, 2)})')

        acc = clf.score(X_test, y_test)
        pred = clf.predict(X_test)
        uar = metrics.balanced_accuracy_score(y_test, pred)
        conf_matrix = metrics.confusion_matrix(y_test, pred)
        test_acc.append(acc)
        test_uar.append(uar)
        conf_matrices.append(conf_matrix)
        print(f'\nCross validation with logistic regression, test accuracy: {acc} (UAR: {uar})')
        print(f'Confusion matrix:\n{conf_matrix}\n')
    print(f'Average test accuracy across {num_folds} folds: {np.mean(test_acc)} (std {np.std(test_acc)})')
    print(f'Average UAR across {num_folds} folds: {np.mean(test_uar)} (std {np.std(test_uar)})')
    print(f'Averaged confusion matrix (rows=target, columns=predicted):\n{np.mean(np.array(conf_matrices), axis=0)}')
    print(f'Sum of all confusion matrices:\n{np.sum(np.array(conf_matrices), axis=0)}')

    if verbose:
        #print(f'\nIntersection of all {num_folds} selected feature sets:\n{set(selected_features[0]).intersection(set(selected_features[1]), set(selected_features[2]), set(selected_features[3]), set(selected_features[4]))}')
        #feats, counts = np.unique(np.array(selected_features), return_counts=True)
        print('\nAll selected features and how many times they had been selected:')
        #for f, c in zip(feats, counts):
            #print(f,c)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='1-Specificity', ylabel='Sensitivity')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, facecolor='white')
    plt.show()
    if return_fold_indices:
        return train_indices, test_indicies


def load_and_preprocess_data(config):
    # load and preprocess data

    DATA = pd.read_csv(
        config['data']['data_file'],
        low_memory=False,
        dtype={'session_info_id_used_by_customer_for_patient': object}
    )
    if 'score_mapping' in config['surveys']:
        survey_score_mapping = json.load(open(config['surveys']['score_mapping']))

    # load MetricsPerTaskReference to get list of metrics
    metrics_per_task_ref = pd.read_csv(config['data']['metrics_per_task_ref'])
    speech_metrics = metrics_per_task_ref[metrics_per_task_ref['type']=='speechMetrics']['metric'].to_list()
    facial_metrics = metrics_per_task_ref[metrics_per_task_ref['type']=='facialMetrics']['metric'].to_list()
    mediapipe_metrics = metrics_per_task_ref[metrics_per_task_ref['type']=='mediapipe_metrics']['metric'].to_list()

    # do some preprocessing
    DATA['session_info_timestamp'] = pd.to_datetime(DATA['session_info_timestamp'], errors='raise')
    DATA['session_info_days_since_first_session_in_dataset'] = (DATA['session_info_timestamp'] - DATA['session_info_timestamp'].min()).dt.days

    DATA['session_info_first_session_timestamp'] = pd.NaT
    if sum(DATA['session_info_access_code'].isna()) > 0:
        logging.info("\nERROR: column session_info_access_code must not contain any NaN.\n\n")
        return None
    for user_id in DATA['session_info_access_code'].unique():
        #print(user_id)
        user_data = DATA[DATA['session_info_access_code']==user_id].sort_values('session_info_timestamp')

        #if len(user_data) == 0:
        #    print(user_data)
        #    continue
        baseline_timestamp = user_data.iloc[0]['session_info_timestamp']
        DATA.loc[DATA['session_info_access_code']==user_id, 'session_info_first_session_timestamp'] = baseline_timestamp
    DATA['session_info_days_since_baseline'] = (DATA['session_info_timestamp'] - DATA['session_info_first_session_timestamp']).dt.days
    # set all volume metrics to NaN between 2021-06-19 and 2021-08-12 (note, this is not needed if metrics come from the research_metrics db table)
    # https://modalityai.atlassian.net/browse/RND-1884
    rows = DATA[(DATA['session_info_timestamp'] < datetime.datetime(2021, 8, 12)) & (DATA['session_info_timestamp'] > datetime.datetime(2021, 6, 19))].index
    cols = DATA.filter(regex='.*volume.*').columns
    DATA.loc[rows, cols] = np.nan
    # drop old openSmile metrics if they are present:
    DATA = DATA.drop(columns=DATA.filter(regex='.*sma_amean|frameTime|frameIndex.*').columns)

    if config['data'].getboolean('aggregate_metrics_across_all_prompts'):
        DATA = aggregate_metrics(DATA, speech_metrics, prefix='speech_metrics')
        DATA = aggregate_metrics(DATA, facial_metrics, prefix='facial_metrics')
        DATA = aggregate_metrics(DATA, mediapipe_metrics, prefix='mediapipe_metrics')

    if 'sessions_to_exclude' in config['data']:
        sessions_to_exclude = ast.literal_eval(config['data']['sessions_to_exclude'])
        if len(sessions_to_exclude) > 0:
            DATA = DATA.drop(DATA[DATA['session_info_session_id'].isin(sessions_to_exclude)].index)
            logging.info(f'Removed the following sessions from the dataset:\n{pprint.pformat(sessions_to_exclude)}')
    if 'access_codes_to_exclude' in config['data']:
        access_codes_to_exclude = ast.literal_eval(config['data']['access_codes_to_exclude'])
        if len(access_codes_to_exclude) > 0:
            DATA = DATA.drop(DATA[DATA['session_info_access_code'].isin(access_codes_to_exclude)].index)
            logging.info(f'Removed all sessions from the following user IDs (access codes):\n{pprint.pformat(access_codes_to_exclude)}')

    # add demographics
    if 'demographics' in config and 'demographics_file' in config['demographics']:
        demographics = pd.read_csv(config['demographics']['demographics_file'], dtype={config['demographics']['user_key']: object})
        # general case: assuming that demographics contains one row per user (APST is an exception)
        # config['demographics']['user_key'] is either 'id_used_by_customer_for_patient' or 'access_code'
        # add prefix 'demo_' to all columns, this makes filtering columns later on easier
        demographics.columns = ['demo_' + x for x in demographics.columns]
        # for NKI, remove leading zeros
        demographics['demo_id_used_by_customer_for_patient'] = demographics['demo_id_used_by_customer_for_patient'].apply(lambda x: re.sub("^0+(?!$)", "", x))
        DATA['session_info_id_used_by_customer_for_patient'] = DATA['session_info_id_used_by_customer_for_patient'].apply(lambda x: re.sub("^0+(?!$)", "", x))
        DATA = pd.merge(
            demographics,
            DATA,
            left_on=f"demo_{config['demographics']['user_key']}",
            right_on=f"session_info_{config['demographics']['user_key']}",
            how='inner', suffixes=('_demo', ''))
        if 'demo_date_of_birth' in DATA.columns:
            DATA['demo_date_of_birth'] = pd.to_datetime(DATA['demo_date_of_birth'], errors='coerce')
            date_of_birth_null_count = sum(DATA['demo_date_of_birth'].isna())
            if date_of_birth_null_count > 0:
                logging.info(f"Demographic information 'date_of_birth' is not available for these participants:\n{DATA[DATA['demo_date_of_birth'].isna()]['session_info_access_code'].unique()}\n")
            if 'ignore_age_threshold' in config['demographics']:
                year_threshold = datetime.datetime.now().year - int(config['demographics']['ignore_age_threshold'])
                DATA['demo_date_of_birth'] = DATA['demo_date_of_birth'].map(lambda x: (x if x.year < (year_threshold) else pd.NaT))
                logging.info(f"When ignoring year of birth >= {year_threshold}, 'date_of_birth' is 'NaT' for these participants:\n{DATA[DATA['demo_date_of_birth'].isna()]['session_info_' + config['demographics']['user_key']].unique()}\n")
            # compute participants' age at timepoint of session
            if 'demo_age' in DATA.columns:
                # if the demographics data already has a column 'age', prefer this information
                DATA['demo_age'] = DATA.apply(lambda x: compute_age(x['demo_date_of_birth'], x['session_info_timestamp']) if pd.isna(x['demo_age']) else x['demo_age'], axis=1)
            else:
                DATA['demo_age'] = DATA.apply(lambda x: compute_age(x['demo_date_of_birth'], x['session_info_timestamp']), axis=1)
        if 'patient_indicator_column' in config['demographics']:
            if not 'patient_indicator_value' in config['demographics']:
                logging.info('WARNING: patient_indicator_value must be specified in demographics section of the config. Treating all subjects as Controls.')
                DATA['demo_patient_flag'] = 0
            else:
                DATA['demo_patient_flag'] = DATA['demo_' + config['demographics']['patient_indicator_column']].apply(lambda x: 1 if x == config['demographics']['patient_indicator_value'] else 0)
        else:
            DATA['demo_patient_flag'] = 0
            logging.info('WARNING: No patient_indicator_column specified in config. Treating all subjects as Controls.')
        # set the default cohorts as patient / controls
        DATA['demo_cohort'] = DATA['demo_patient_flag']
        DATA['demo_cohort_name'] = DATA['demo_patient_flag'].apply(lambda x: 'Control' if x == 0 else 'Patient')
    else:
        logging.info('No demographic information specified in the config file.')
    DATA = DATA.set_index('session_info_session_id')
    return DATA


def plot_data_distributions(dataframe, categories, hue=None, colors=None, show_probplot=False):
    """Plots histograms for all dataframe columns of a specified category.

    Parameters:
        dataframe (pandas DataFrame): data, must contain the columns of the categories of interest
        categories (list of str): contains column prefixes to filter for, e.g. 'speech_metrics'
        hue (str): (optional) column containing groups for plotting separate, overlaid histograms
        colors (list): (optional) a list of Matplotlib color codes, needs to contain at least as many colors as groups found in 'hue' column

    Returns:
        No return value.
    """
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.keys())
    if colors is not None and hue is not None:
        if len(colors) < len(dataframe[hue].unique()):
            print("Argument colors must contain at least as many color names as there are distinct values in the 'hue' column")
            return False
    def print_stats(data_series):
        print(data_series.aggregate(['count', 'mean', 'median', 'std', 'max', 'min']))
        print(f"Skewness: {stats.skew(data_series, nan_policy='omit')}\nKurtosis: {stats.kurtosis(data_series, nan_policy='omit')}\n")
    plt.close()
    for category in categories:
        for metric in dataframe.filter(regex=category).columns:
            if pd.isnull(dataframe[metric]).all():
                print(f"Skipping metric {metric} because all values are NaN")
                continue
            if hue is not None:
                continue_flag = False
                for group in dataframe[hue].unique():
                    if pd.isnull(dataframe[dataframe[hue]==group][metric]).all():
                        print(f"Skipping metric {metric} because all values are NaN for group {group}")
                        continue_flag = True
                if continue_flag:
                    continue
            if show_probplot:
                num_axes = 2 if hue is None else len(dataframe[hue].unique()) + 1
                fig, ax = plt.subplots(1, num_axes)
                fig.set_size_inches(6 * num_axes, 4)
            else:
                fig, ax = plt.subplots(1, 1)
                ax = [ax]
            if hue is None:
                print_stats(dataframe[metric])
                ax[0].hist(dataframe[metric], color='#CC6633', alpha=1, label=metric)
                if show_probplot:
                    # dropna() is applied here because otherwise the linear fit would not be done and plotted as a reference
                    stats.probplot(dataframe[metric].dropna(), plot=ax[1])
            else:
                for index, group in enumerate(dataframe[hue].unique()):
                    print(f"Group {group}:")
                    print_stats(dataframe[dataframe[hue]==group][metric])
                    ax[0].hist(dataframe[dataframe[hue]==group][metric], color=colors[index], alpha=0.5, label=str(group))
                    if show_probplot:
                        # dropna() is applied here because otherwise the linear fit would not be done and plotted as a reference
                        stats.probplot(dataframe[dataframe[hue]==group][metric].dropna(), plot=ax[index + 1])
                        ax[index + 1].set_title(f"Prob plot - {group}")
            ax[0].legend()
            plt.show()


def test_normality_of_distribution(dataframe, categories, report_all=False, summary_only=False, alpha=0.05):
    """Statistical tests for normality of data distributions.

    This function computes the following three tests on all columns in the given categories: skewness, kurtosis, Shapiro-Wilk.

    Parameters:
        dataframe (pandas DataFrame): data, must contain the columns of the categories of interest
        categories (list of str): contains column prefixes to filter for, e.g. 'speech_metrics'
        report_all (bool): if True, report all test outcomes, otherwise only the ones from non-normal distributions. Default: False
        summary_only (bool): if True, no test statistics are printed, but only the number of columns that seem not normally distributed. Default: False
        alpha: required level for statistical significance. Default: 0.05

    Returns:
        No return value.

    """
    for category in categories:
        non_normal_count = 0
        skip_count = 0
        for metric in dataframe.filter(regex=category).columns:
            if len(dataframe[metric].dropna()) < 8:
                if not summary_only:
                    print(f"Could not compute statistics for {metric} because it has less than 8 samples.\n")
                skip_count += 1
                continue
            # skewtest, kurtosistest (normality tests)
            # This function tests the null hypothesis that the skewness of the population that the sample was drawn from is the same as that of a corresponding normal distribution.
            skewtest = stats.skewtest(dataframe[metric].dropna())
            # This function tests the null hypothesis that the kurtosis of the population from which the sample was drawn is that of the normal distribution.
            kurtosistest = stats.kurtosistest(dataframe[metric].dropna())
            # Shapiro-Wilk test of normality
            # null hypothesis: data is normally distributed; reject if p < chosen_alpha, e.g. p < 0.05
            shapiro = stats.shapiro(dataframe[metric].dropna())

            if skewtest[1] < alpha or kurtosistest[1] < alpha or shapiro[1] < alpha:
                non_normal_count += 1
            if (skewtest[1] < alpha or kurtosistest[1] < alpha or shapiro[1] < alpha or report_all) and not summary_only:
                print(f"{metric} test statistics:")
                print(f"\tSkew test statistic: {np.round(skewtest[0], 3)}, pvalue: {np.round(skewtest[1], 3)}")
                print(f"\tKurtosis test statistic: {np.round(kurtosistest[0], 3)}, pvalue: {np.round(kurtosistest[1], 3)}")
                print(f"\tShapiro-Wilk statistic: {np.round(shapiro[0], 3)}, pvalue: {np.round(shapiro[1], 3)}\n")
        print(f"\nFor {category}, {non_normal_count} out of {len(dataframe.filter(regex=category).columns)} columns do NOT represent normally distributed data. {skip_count} columns were skipped.\n")


def kolmogorov_smirnov(dataframe, categories, cohort_names):
    """Non-parametric test of the equality of two distributions.

    Parameters:
        dataframe (pandas DataFrame): data, must contain the columns of the categories of interest
        categories (list of str): contains column prefixes to filter for, e.g. 'speech_metrics'
        cohort_names (list of str): must contain two names of cohorts that occur in dataframe['demo_cohort_name']

    Returns:
        No return value.

    """
    if len(cohort_names) != 2:
        print("Cannot do the Kolmogorov-Smirnov test. Cohort_names must contain exactly TWO elements.")
        return None
    for category in categories:
        for metric in dataframe.filter(regex=category).columns:
            data_0 = dataframe[dataframe['demo_cohort_name']==cohort_names[0]][metric].dropna()
            data_1 = dataframe[dataframe['demo_cohort_name']==cohort_names[1]][metric].dropna()
            if len(data_0) < 20 or len(data_1) < 20:
                print(f"Skipping {metric} because sample size of at least one distribution is less than 20.")
                continue
            # The null hypothesis is that the two distributions are identical
            d, p = stats.kstest(data_0,
                                data_1,
                                alternative='two-sided')
            print(f"\n{metric} Kolmogorov-Smirnov test statistic: {np.round(d, 3)}, pvalue: {np.round(p, 3)}\n")
            # plot Cumulative distribution function (CDF)
            sns.kdeplot(data = data_0, cumulative = True, label=f"{cohort_names[0]} (n={len(data_0)})")
            sns.kdeplot(data = data_1, cumulative = True, label=f"{cohort_names[1]} (n={len(data_1)})")
            plt.legend()
            plt.show()
    #n = len(dataframe[dataframe['demo_cohort_name']==cohort_names[0]][metric].dropna())
    #m = len(dataframe[dataframe['demo_cohort_name']==cohort_names[1]][metric].dropna())
    #print(f"Reject null hypothesis at p=0.05: {d > 1.358 * np.sqrt((n + m) / (n * m))}")
    #print(f"D={np.round(d, 4)}, p={np.round(p, 4)}\n")


def report_missing_values(dataframe, categories):
    """For given data categories, report number of missing values per column.

    Parameters:
        dataframe (pandas DataFrame): data, must contain the columns of the categories of interest
        categories (list of str): contains column prefixes to filter for, e.g. 'speech_metrics'

    Returns:
        No return value.

    """
    for category in categories:
        metrics_data = dataframe.filter(regex=category)
        print(f"\nMissing {category} metrics/information:")

        # report rows and columns where all data is NaN
        cols_all_nan = metrics_data.columns[metrics_data.isna().all()]
        if cols_all_nan.empty:
            print('All columns contain at least some data.')
        else:
            print(f"Columns where ALL values are NaN: {cols_all_nan}")

        rows_all_nan = metrics_data.index[metrics_data.isna().all(axis=1)]
        if rows_all_nan.empty:
            print('All rows contain at least some data.')
        else:
            print(f"Rows where ALL values are NaN: {rows_all_nan}")

        # for each metric, report number of NaNs (if greater than 0)
        print()
        missing_values = {}
        cols_with_nans = metrics_data.columns[metrics_data.isna().any()].to_list()
        for col in cols_with_nans:
            missing_values[col] = sum(metrics_data[col].isna())
        print('Number of missing values per column (descending):')
        print(pprint.pformat({k: v for k, v in sorted(missing_values.items(),
                                               key=lambda item: item[1],
                                               reverse=True)}, sort_dicts=False))


def remove_missing_values(dataframe, percentage_threshold, subset=None, verbose=False):
    """Remove missing values from the dataframe.

    This function first removes all columns that have more than (len(dataframe) * percentage_threshold) missing values.
    Then, all rows with any missing values are removed. percentage_threshold can be seen as a tradeoff between how many columns (metrics) and how many samples should be retained.

    Parameters:
        dataframe (pandas DataFrame): Input data.
        percentage_threshold (float): percentage, between 0 and 1.
        subset (list of str): columns to consider for removing missing values, if None, the complete dataframe is considered.
        verbose (bool): if True, print the list of removed metric columns.

    Returns:
        dataframe with missing values removed.
    """
    threshold = len(dataframe) * percentage_threshold
    columns_to_drop = {}
    columns_to_consider = dataframe.columns if subset is None else dataframe[subset].columns
    for column in columns_to_consider:
        if sum(dataframe[column].isna()) > threshold:
            columns_to_drop[column] = sum(dataframe[column].isna())
    if verbose:
        print(f"Removed the following columns from the dataframe because they have more than {threshold} missing values:")
        print(pprint.pformat(columns_to_drop))
    new_dataframe = dataframe.drop(columns=columns_to_drop.keys())

    # drop rows that contain NaNs
    subset = list(set(subset).difference(set(columns_to_drop.keys())))
    new_dataframe = new_dataframe.dropna(axis=0, subset=subset)
    print(f"\nSize of dataframe after removing NaNs: {new_dataframe.shape}\n")

    # after removing data, make sure that 'session_info_days_since_baseline' is recomputed (because it can happen that a user's initial baseline session was dropped)
    for user_id in new_dataframe['session_info_access_code'].unique():
        user_data = new_dataframe[new_dataframe['session_info_access_code']==user_id].sort_values('session_info_timestamp')
        baseline_timestamp = user_data.iloc[0]['session_info_timestamp']
        new_dataframe.loc[new_dataframe['session_info_access_code']==user_id, 'session_info_first_session_timestamp'] = baseline_timestamp
    new_dataframe['session_info_days_since_baseline'] = (new_dataframe['session_info_timestamp'] - new_dataframe['session_info_first_session_timestamp']).dt.days
    return new_dataframe


def translate(text, conversion_dict):
    if not text:
        return text
    for key, value in conversion_dict.items():
        text = text.replace(key, value)
    return text

def convert_strings_to_arrays(dataframe: pd.DataFrame, regex: str) -> pd.DataFrame:
    for col in dataframe.filter(regex=regex, axis=1):
        for i in range(len(dataframe[col])):
            if type(dataframe[col][i]) == str:
                dataframe[col][i] = np.asarray(pd.eval(dataframe[col][i]))
            else:
                dataframe[col][i] = np.full(shape = 49, fill_value=nan)
    return dataframe

def reshape_timeseries_dataframe(dataframe: pd.DataFrame, regex: str) -> pd.DataFrame:
    features = dataframe.filter(regex=regex).columns.to_list()

    length_timeseries = dataframe[features[0]][0].size
    for feature in dataframe[features]:
        for array in dataframe[feature]:
            assert  array.size == length_timeseries, f"Array sizes vary within dataframe! Current size is {array.size} for feature '{feature}', previously seen size {length_timeseries}."

    print(f"length_timeseries: {length_timeseries}")

    X_flattened = {}
    for feature in dataframe[features]:
        for array in dataframe[feature]:
            is_nan = array[0] == nan
            for idx in range(length_timeseries):
                key = feature + "_" + str(idx)
                val = nan if is_nan else array[idx]
                if key in X_flattened:
                    X_flattened[key].append(val)
                else:
                    X_flattened[key] = [val]
    dataframe_flattened = pd.DataFrame(X_flattened, index = dataframe.index)
    for col in dataframe:
        if col not in features:
            dataframe_flattened[col] = dataframe[col].tolist()

    return dataframe_flattened
