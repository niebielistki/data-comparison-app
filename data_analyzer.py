#! python3
# data_analyzer.py

import pandas as pd
from csv_handler import CSVHandler
from scipy.stats import pearsonr
from PyQt5.QtCore import QObject, pyqtSignal
from collections import defaultdict
import numpy as np
import traceback
from sklearn.linear_model import LinearRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string



class DataAnalyzer(QObject):
    analysisComplete = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.csv_handler = CSVHandler()


    # Templates for Textual Analysis
    def get_analysis_templates(self, analysis_type, data_type):
        """
        Returns the appropriate analysis templates based on the analysis and data type.
        :param analysis_type: A string indicating the type of analysis ('yearly', 'monthly', 'daily', 'data_cleaning', 'descriptive_stats').
        :param data_type: A string indicating the data type ('numeric', 'text', 'both').
        :return: A list of templates as strings.
        """
        templates = {
            'yearly': {
                'numeric': ["A comparison from {start_year} to {end_year} shows {variable} fluctuating, with a net change of {value}.",
                            "Year-over-year, {variable} demonstrated a {trend} trend.",
                            "The year {year1} marked a significant {trend} in {variable}, deviating from previous trends.",
                            "A steady {trend} in {variable} was observed every year from {start_year} to {end_year}.",
                            "The largest year-on-year change in {variable} was between {year1} and {year2}, showing a {percentage}% difference.",
                            "The cumulative change in {variable} over the last {number} years was {value}, indicating a long-term {trend}.",
                            "An alternating pattern of {trend} in {variable} was noted every alternate year from {start_year} to {end_year}.",
                            "The {variable} in {year1} was surprisingly {higher/lower} than the preceding {number} years' average, marking an anomaly."
                ],
                'text': ["In {month}, the {column_name} showed a notable {trend} in {variable}.",
                         "The {column_name} experienced its peak in {month/year}, with significant {metrics}.",
                         "During {month/year}, a shift towards {column_name} was observed, influencing {variable} considerably.",
                         "In {month/year}, {column_name} marked a turning point, with a marked {trend} in {variable}.",
                         "A consistent trend in {column_name} across consecutive {months/years} indicates a stable market for {variable}.",
                         "Seasonal variations are evident in {column_name}, with {month} typically showing a distinct pattern in {variable}.",
                         "Comparative analysis reveals that {month/year} was pivotal for {column_name}, diverging from previous trends in {variable}.",
                         "Anomaly detection in {month/year} for {column_name} suggests an unusual pattern in {variable}, warranting further investigation."
                ]
            },
            'monthly': {
                'numeric': ["In {month}, {variable} tends to show a {rise/drop}, relative to previous months.",
                            "The month of {month} is historically known for the highest/lowest {variable}, averaging {value}.",
                            "A recurring {increase/decrease} in {variable} is evident during the {month(s)}.",
                            "Comparing months, {variable} in {month1} is often {percentage}% {higher/lower} than in {month2}.",
                            "The {month} period typically sees a range of {variable} from {lowest value} to {highest value}.",
                            "Over the last {number} years, {variable} has shown a gradual {increase/decrease} during the {month} period.",
                            "The variance of {variable} in {month} has been decreasing/increasing, suggesting a stabilization/volatility over the years.",
                            "In comparing {month1} to {month2}, a consistent shift in the pattern of {variable} is observed, with {month2} often showing a contrasting trend."
                ],
                'text': ["In {month}, the {column_name} showed a notable {trend} in {variable}.",
                         "The {column_name} experienced its peak in {month/year}, with significant {metrics}.",
                         "During {month/year}, a shift towards {column_name} was observed, influencing {variable} considerably.",
                         "In {month/year}, {column_name} marked a turning point, with a marked {trend} in {variable}.",
                         "A consistent trend in {column_name} across consecutive {months/years} indicates a stable market for {variable}.",
                         "Seasonal variations are evident in {column_name}, with {month} typically showing a distinct pattern in {variable}.",
                         "Comparative analysis reveals that {month/year} was pivotal for {column_name}, diverging from previous trends in {variable}.",
                         "Anomaly detection in {month/year} for {column_name} suggests an unusual pattern in {variable}, warranting further investigation."
                ]
            },
            'daily': {
                'numeric': ["On {specific_date}, a record-breaking {variable} value of {value} was observed.",
                            "There was a notable shift in {variable} starting from {start_date}, persisting until {end_date}.",
                            "Consecutive days from {start_date} to {end_date} showed a consistent {increase/decrease} in {variable}.",
                            "The period between {start_date} and {end_date} witnessed the most volatility in {variable}, with daily changes of up to {value}.",
                            "A pattern emerged where {variable} consistently peaked around {specific_date(s)}.",
                            "The {variable} showed an unexpected spike/drop on {specific_date}, deviating significantly from the monthly average.",
                            "Week-to-week analysis from {start_date} to {end_date} reveals a consistent {increase/decrease} in {variable}.",
                            "A sudden change in {variable} was observed around {specific_date}, marking a distinct shift from the established pattern."
                ],
                'text': ["On {specific_date}, a notable {trend} was observed in {variable} within the {column_name}.",
                         "A significant event in {column_name} on {specific_date} led to a marked change in {variable}.",
                         "The data shows an unusual spike/drop in {variable} on {specific_date}, indicating an outlier or a special event in {column_name}.",
                         "Sequential days around {specific_date} reveal a consistent pattern in {variable}, highlighting trends in {column_name}.",
                         "A comparison of {variable} on {specific_date} against the same date in previous years shows a {trend} in {column_name}.",
                         "The period immediately following {specific_date} shows a noticeable shift in {variable}, reflecting changes in {column_name}.",
                         "On {specific_date}, {column_name} reached a historical high/low in {variable}, setting a new record",
                         "The week of {specific_date} was critical for {column_name}, with {variable} showing significant fluctuations."
                ]
            },
            'data_cleaning': {
                'numeric': ["Missing Data: {column_name} column has {percentage_missing_data}% missing data.",
                            "Data Type Inconsistency: {column_name} has {number_of_inconsistencies} entries with non-numeric values.",
                            "Outlier Detection: {column_name} column has {number_of_outliers} potential outliers (values significantly higher/lower than the rest).",
                            "Duplicate Data: {number_of_duplicate_entries} duplicate entries found.",
                            "Invalid Data: {column_name} has {number_of_invalid_entries} entries with invalid values (e.g., negative where only positive expected)."
                ],
                'text': ["Missing Data: {column_name} column has {percentage_missing_data}% missing data.",
                         "Text Length Analysis: {column_name} column has several unusually long/short entries.",
                         "Missing Text Data: {column_name} column missing in {number_of_missing_rows} rows.",
                         "Average Word Length per Column: {column_name_1}: {average_length_1} characters, {column_name_2}: {average_length_2} characters, {column_name_3}: {average_length_3} characters."
                ]
            },
            'descriptive_stats': {
                'numeric': ["Mean: Average '{column_name}' ({start_year}-{end_year}) - {average_value}.",
                            "Median: Median '{column_name}' ({start_year}-{end_year}) - {median_value}.",
                            "Mode: Most common '{column_name}' - {mode_value}.",
                            "Range: '{column_name}' Range ({start_year}-{end_year}) - {lowest_value} to {highest_value}.",
                            "Variance and Standard Deviation: Variance in '{column_name}' - {variance_value}. Standard Deviation - {standard_deviation_value}.",
                            "Skewness: '{column_name}' shows a {type_of_skew} skew (positive/negative skewness).",
                            "Kurtosis: '{column_name}' shows a {type_of_kurtosis} distribution (e.g., high/low kurtosis).",
                            "Percentiles/Quartiles: Key percentiles of '{column_name}' - 25th: {25th_percentile_value}; 50th: {50th_percentile_value}; 75th: {75th_percentile_value}.",
                            "Correlation Matrix: Relationship between '{column_name1}' and '{column_name2}' shows a {type_of_correlation} correlation."
                ],
                'text': ["Most frequent keyword in {column_name_1}: {most_frequent_keyword_1} (appeared {count_1} times).",
                         "Least appearing keyword in {column_name_1}: {least_frequent_keyword_1} (appeared {count_least_1} times).",
                         "Most frequent keyword in {column_name_2}: {most_frequent_keyword_2} (appeared {count_2} times).",
                         "Least appearing keyword in {column_name_2}: {least_frequent_keyword_2} (appeared {count_least_2} times).",
                         "Most frequent keyword in {column_name_3}: {most_frequent_keyword_3} (appeared {count_3} times).",
                         "Least appearing keyword in {column_name_3}: {least_frequent_keyword_3} (appeared {count_least_3} times)."
                ]
            }
        }

        selected_templates = templates[analysis_type][data_type]
        if data_type == 'both':
            # Combine both numeric and text templates
            selected_templates = templates[analysis_type]['numeric'] + templates[analysis_type]['text']

        return selected_templates

    def fill_placeholders(self, template, placeholder_values):
        """
        Fills placeholders in a template string with provided values.

        :param template: A string template with placeholders.
        :param placeholder_values: A dictionary where keys are placeholders in the template and values are the actual values to replace.
        :return: A string with placeholders filled with actual values.
        """
        try:
            return template.format(**placeholder_values)
        except KeyError as e:
            print(f"Missing placeholder value: {e}")
            return template

    # Function for Time Series Analysis
    def analyze_time_series(self, data_frames, time_series_priority=None):
        analysis_results = []

        # If a priority is given, use it. Otherwise, determine it based on data
        if not time_series_priority:
            time_series_priority = self.determine_priority_based_on_data(data_frames)

        # Convert the priority to a list for consistency with existing code
        time_priority = [time_series_priority] if time_series_priority else sorted(
            self.detect_time_columns(data_frames).keys(), key=lambda k: k[1], reverse=True)

        for file_name, df in data_frames.items():
            # Detect time columns with occurrence count
            time_columns = self.detect_time_columns(df)

            # Priority is given to numerical columns for analysis
            analysis_variable = next((col for col in df.columns if df[col].dtype in [np.float64, np.int64]), None)

            # Perform analysis based on priority
            for time_type in time_priority:
                column_name = time_columns.get(time_type)
                if column_name and analysis_variable:
                    if time_type == 'Year':
                        analysis_results.append(self._analyze_yearly_data(df, column_name, analysis_variable))
                    elif time_type == 'Month':
                        analysis_results.append(self._analyze_monthly_data(df, column_name, analysis_variable))
                    elif time_type == 'Date':
                        analysis_results.append(self._analyze_daily_data(df, column_name, analysis_variable))

            # Append results from data cleaning tools and descriptive statistics
            analysis_results.append(self.perform_data_cleaning_analysis(df))
            descriptive_stats = self.calculate_descriptive_statistics(df)
            if not descriptive_stats.empty:
                analysis_results.append(descriptive_stats.to_string(index=False))

        return "\n\n".join(analysis_results)

    # Helper function for analyze_time_series
    def detect_time_columns(self, data_frames):
        """
        Detects columns in multiple DataFrames that are likely to contain time-related data and counts their occurrences.
        :param data_frames: Dictionary of DataFrames to analyze.
        :return: Dictionary with time column types as keys and tuples (column name, occurrence count) as values.
        """
        time_columns = defaultdict(lambda: defaultdict(int))
        for _, df in data_frames.items():  # Explicitly iterate over dictionary items
            if isinstance(df, pd.DataFrame):  # Check if each item is a DataFrame
                for col in df.columns:
                    if col.lower() in ['year', 'years']:
                        time_columns['Year'][col] += 1
                    elif col.lower() in ['month', 'months']:
                        time_columns['Month'][col] += 1
                    elif col.lower() in ['date', 'dates']:
                        time_columns['Date'][col] += 1

        # Summarizing results to get the most common column for each time type
        summarized_time_columns = {}
        for time_type, columns in time_columns.items():
            if columns:
                most_common_column = max(columns, key=columns.get)
                summarized_time_columns[time_type] = (most_common_column, sum(columns.values()))

        return summarized_time_columns

    def determine_priority_based_on_data(self, data_frames):
        time_columns = self.detect_time_columns(data_frames)
        # Sort the time columns based on completeness or other criteria
        sorted_time_columns = sorted(time_columns.items(), key=lambda item: item[1][1], reverse=True)
        # Return the type of the time column with the highest priority
        return sorted_time_columns[0][0] if sorted_time_columns else None

    def _analyze_yearly_data(self, df, year_column, variable):
        """
        Analyzes yearly data for a given variable.
        :param df: DataFrame containing the data.
        :param year_column: Column name with year data.
        :param variable: The variable to be analyzed.
        :return: Textual analysis for yearly data.
        """
        if variable not in df.columns:
            return "No data for variable '{}' to analyze.".format(variable)

        try:
            # Ensure year_column is in datetime format and set it as index
            df[year_column] = pd.to_datetime(df[year_column], format='%Y', errors='coerce')
            df.set_index(year_column, inplace=True)

            # Sort by year and calculate differences
            df_sorted = df.sort_values(by=year_column)
            df_sorted['yearly_change'] = df_sorted[variable].diff()

            # Volatility calculation over a rolling 12-month period
            df_sorted['yearly_volatility'] = df_sorted[variable].rolling('365D').std()
            most_volatile_year = df_sorted['yearly_volatility'].idxmax().year
            least_volatile_year = df_sorted['yearly_volatility'].idxmin().year

            # Calculate the net change
            net_change = df_sorted[variable].iloc[-1] - df_sorted[variable].iloc[0]
            fluctuation = "fluctuating" if net_change != 0 else "stable"
            trend = "increasing" if net_change > 0 else "decreasing" if net_change < 0 else "stable"

            # Find the largest year-on-year change
            max_change = df_sorted['yearly_change'].abs().max()
            max_change_years = df_sorted[df_sorted['yearly_change'].abs() == max_change][year_column].values
            max_change_percentage = (max_change / df_sorted[variable].abs().mean()) * 100

            # Check for alternating patterns
            alternating_years = df_sorted[df_sorted['yearly_change'].abs() > 0][year_column].tolist()
            alternating_pattern = "alternating" if len(alternating_years) % 2 == 0 else "not alternating"

            # Compare each year with the previous years' average
            df_sorted['cumulative_average'] = df_sorted[variable].expanding().mean()
            current_year = df_sorted[year_column].iloc[-1]
            current_value = df_sorted[variable].iloc[-1]
            anomaly = current_value > df_sorted['cumulative_average'].iloc[-2]

            # Prepare data for template-based analysis
            analysis_data = {
                'start_year': start_year,
                'end_year': end_year,
                'net_change': net_change,
                # ... [other data points] ...
            }

            # Get the appropriate analysis templates
            analysis_templates = self.get_analysis_templates('yearly', 'numeric')

            # Use the templates to build the analysis text
            analysis_text = [template.format(**analysis_data) for template in analysis_templates]

            # Add volatility analysis
            # ... [volatility analysis code] ...

            return "\n".join(analysis_text)

        except Exception as e:
            return f"Error in analyzing yearly data: {e}"

    def _analyze_monthly_data(self, df, month_column, variable):
        """
        Analyzes monthly data for a given variable.
        :param df: DataFrame containing the data.
        :param month_column: Column name with month data.
        :param variable: The variable to be analyzed.
        :return: Textual analysis for monthly data.
        """
        if variable not in df.columns:
            return "No data for variable '{}' to analyze.".format(variable)

        try:
            df[month_column] = pd.to_datetime(df[month_column], errors='coerce')
            if df[month_column].isnull().any():
                return "Invalid or missing month data. Cannot perform analysis."

            df['Month'] = df[month_column].dt.month
            df['Year'] = df[month_column].dt.year
            df.set_index(month_column, inplace=True)

            # Calculate and append volatility analysis
            df['monthly_volatility'] = df[variable].rolling('30D').std()
            most_volatile_month = df['monthly_volatility'].idxmax()
            least_volatile_month = df['monthly_volatility'].idxmin()

            # Prepare data for template-based analysis
            analysis_data = {
                'most_volatile_month': most_volatile_month.strftime('%Y-%m'),
                'least_volatile_month': least_volatile_month.strftime('%Y-%m'),
                # Additional data points required for the templates
                'month': df.index.month_name().unique()[0],  # Assumes analysis is for a specific month
                'variable': variable,
                'rise_drop': 'rise' if df[variable].diff().mean() > 0 else 'drop',
                'highest_value': df[variable].max(),
                'lowest_value': df[variable].min(),
                'average_value': df[variable].mean(),
                'percentage_change': df[variable].pct_change().mean() * 100,  # Average percentage change
                'increase_decrease': 'increase' if df[variable].diff().mean() > 0 else 'decrease',
                'number_of_years': df['Year'].nunique(),
                'trend': 'increasing' if df[variable].diff().mean() > 0 else 'decreasing',
                'stabilization_volatility': 'stabilization' if df['monthly_volatility'].mean() < df[
                    variable].std() else 'volatility',
                # ... [any other specific data points required by your templates] ...
            }

            # Get the appropriate analysis templates
            analysis_templates = self.get_analysis_templates('monthly', 'numeric')

            # Use the templates to build the analysis text
            analysis_text = [template.format(**analysis_data) for template in analysis_templates]

            return "\n".join(analysis_text)

        except Exception as e:
            return "Error in analyzing monthly data: {}".format(e)

    def _analyze_daily_data(self, df, date_column, variable):
        if variable not in df.columns:
            return f"No data for variable '{variable}' to analyze."

        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            if df[date_column].isnull().any():
                return "Invalid or missing date data. Cannot perform analysis."

            df = df.sort_values(by=date_column)

            # Prepare data for analysis template
            analysis_data = {
                'specific_date': df[date_column].max().strftime('%Y-%m-%d'),  # Example for template
                'start_date': df[date_column].min().strftime('%Y-%m-%d'),
                'end_date': df[date_column].max().strftime('%Y-%m-%d'),
                'variable': variable,
                'record_value': df[variable].max(),
                'record_date': df[df[variable] == df[variable].max()][date_column].dt.strftime('%Y-%m-%d').iloc[0],
                'max_change': df[variable].diff().abs().max(),
                'shift_start_date':
                    df[df[variable].diff().abs() == df[variable].diff().abs().max()][date_column].dt.strftime(
                        '%Y-%m-%d').iloc[0],
                'shift_end_date': (df[df[variable].diff().abs() == df[variable].diff().abs().max()][
                                       date_column] + pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d').iloc[0],
                'consecutive_increase': df[variable].diff().fillna(0) > 0,
                'consecutive_decrease': df[variable].diff().fillna(0) < 0,
                'rolling_volatility': df[variable].rolling('7D').std(),  # 7-day rolling window for volatility
                'most_volatile_date': df['rolling_volatility'].idxmax().strftime('%Y-%m-%d'),
                'least_volatile_date': df['rolling_volatility'].idxmin().strftime('%Y-%m-%d'),
                'highest_volatility_value': df['rolling_volatility'].max()
                # ... [additional data points] ...
            }

            # Calculate volatility and add to analysis_data
            df['daily_volatility'] = df[variable].rolling('7D').std()
            analysis_data['most_volatile_date'] = df['daily_volatility'].idxmax().strftime('%Y-%m-%d')
            analysis_data['least_volatile_date'] = df['daily_volatility'].idxmin().strftime('%Y-%m-%d')
            analysis_data['highest_volatility_value'] = df['daily_volatility'].max()

            # Get the appropriate analysis templates for daily numeric data
            analysis_templates = self.get_analysis_templates('daily', 'numeric')

            # Use the templates to build the analysis text
            analysis_text = [template.format(**analysis_data) for template in analysis_templates]

            return "\n".join(analysis_text)
        except Exception as e:
            return f"Error in analyzing daily data: {e}"

    # Function for Data Cleaning Tools
    def perform_data_cleaning_analysis(self, df):
        """
        Performs analysis on the dataframe to identify and report data issues.
        :param df: DataFrame containing the data.
        :return: String with results from data cleaning analysis.
        """
        analysis_text = ["Data Cleaning Tools:"]

        # Missing Data Identification
        missing_data = df.isnull().sum()
        missing_data_percentage = (missing_data / len(df)) * 100
        for column, missing in missing_data.items():
            if missing > 0:
                analysis_text.append(
                    f'- Missing Data: "{column}" column has {missing_data_percentage[column]:.2f}% missing data.')

        # Data Type Inconsistency
        for column in df.columns:
            data_types = df[column].apply(lambda x: type(x).__name__).value_counts()
            if len(data_types) > 1:
                analysis_text.append(f'- Data Type Inconsistency: "{column}" has {data_types.to_dict()}.')

        # Outlier Detection
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
            if not outliers.empty:
                analysis_text.append(
                    f'- Outlier Detection: "{column}" column has {len(outliers)} potential outliers (values significantly higher or lower than the rest).')

        # Duplicate Data
        duplicate_rows = df[df.duplicated()]
        duplicate_text = f'No duplicate entries found.' if duplicate_rows.empty else f'{duplicate_rows.shape[0]} duplicate rows found.'
        analysis_text.append(f'- Duplicate Data: {duplicate_text}')

        # Invalid Data
        # This should be adjusted based on the specific requirements of what constitutes 'invalid' data in your dataset
        for column in df.select_dtypes(include=[np.number]).columns:
            invalid_data = df[df[column] < 0]
            if not invalid_data.empty:
                analysis_text.append(
                    f'- Invalid Data: "{column}" has {invalid_data.shape[0]} entries with negative values.')

        return "\n".join(analysis_text)

    # Function for Descriptive Statistics
    def calculate_descriptive_statistics(self, df):
        """
        Calculates descriptive statistics for numerical columns in a dataframe.
        :param df: DataFrame containing the data.
        :return: DataFrame with descriptive statistics.
        """
        # Filter to only include numerical columns
        numerical_df = df.select_dtypes(include=[np.number])

        if numerical_df.empty:
            print("No numerical data for descriptive statistics.")
            return pd.DataFrame()

        print("Input DataFrame for Descriptive Statistics:\n", numerical_df)

        # Creating a DataFrame to store statistics
        stats_df = pd.DataFrame()

        for column in numerical_df.columns:
            stats = numerical_df[column].describe()
            skewness = numerical_df[column].skew()
            kurtosis = numerical_df[column].kurtosis()

            stats_df[column] = [
                stats['mean'],
                stats['50%'],
                numerical_df[column].mode().iloc[0] if not numerical_df[column].mode().empty else np.nan,
                stats['max'] - stats['min'],
                numerical_df[column].var(),
                stats['std'],
                skewness,
                kurtosis
            ]

        stats_df.index = ['Mean', 'Median', 'Mode', 'Range', 'Variance', 'Standard Deviation', 'Skewness', 'Kurtosis']

        # Correlation Matrix (if there are at least two numerical columns)
        if len(numerical_df.columns) > 1:
            correlation_matrix = numerical_df.corr()
            # Use pd.concat to add the correlation matrix as a new section in the stats_df
            stats_df = pd.concat([stats_df, correlation_matrix], keys=['Descriptive Stats', 'Correlation Matrix'])

        print("Output DataFrame for Descriptive Statistics:\n", stats_df)

        return stats_df

    def perform_detailed_textual_analysis(self, df):
        """
        Performs detailed textual analysis on text columns of a DataFrame.
        Returns a list of analysis sections with 'title' and 'content' (HTML formatted).
        """
        analysis_sections = []
        for column in df.columns:
            if df[column].dtype == 'object':  # Assuming text columns are of object type
                analysis = self.perform_textual_analysis_on_column(df[column])
                if analysis:
                    section = {
                        'title': f"Textual Analysis for {column}",
                        'content': analysis.replace('\n', '<br>')  # Convert newlines to HTML line breaks
                    }
                    analysis_sections.append(section)
        return analysis_sections

    def perform_textual_analysis_on_column(self, column_data):
        """
        Performs textual analysis on a given column of a DataFrame.
        :param column_data: A pandas Series representing a single column of text data.
        :return: String with results of textual analysis.
        """
        if column_data.empty:
            return "No data available for analysis."

        analysis_text = [f"Textual Analysis for Column: {column_data.name}"]

        # Frequency analysis of unique values
        value_counts = column_data.value_counts()
        most_common_value = value_counts.idxmax()
        most_common_count = value_counts.max()

        analysis_text.append(f"Most common value: '{most_common_value}' ({most_common_count} occurrences)")
        for value, count in value_counts.items():
            analysis_text.append(f"'{value}': {count} occurrences")

        return "\n".join(analysis_text)

    def classify_and_analyze_data(self, data_frames):
        print("classify_and_analyze_data method called.")
        try:
            numerical_analysis_results = []
            textual_analysis_sections = []

            for file_path, df in data_frames.items():
                # Perform data cleaning and descriptive statistics analysis
                data_cleaning_text = self.perform_data_cleaning_analysis(df)
                if data_cleaning_text:
                    textual_analysis_sections.append({'title': 'Data Cleaning Tools', 'content': data_cleaning_text})

                descriptive_stats_text = self.calculate_descriptive_statistics(df).to_string()
                if descriptive_stats_text:
                    textual_analysis_sections.append({'title': 'Descriptive Statistics', 'content': descriptive_stats_text})

                # Process each column for numerical analysis
                for column in df.columns:
                    if df[column].dtype in [np.int64, np.float64]:
                        print(f"Processing numerical column: {column}")
                        numerical_df = df[[column]]
                        stats_df = self.calculate_descriptive_statistics(numerical_df)
                        if not stats_df.empty:
                            numerical_analysis_results.append(stats_df)

                # Detailed textual analysis
                detailed_textual_sections = self.perform_detailed_textual_analysis(df)
                textual_analysis_sections.extend(detailed_textual_sections)

            # Concatenate numerical analysis results
            if numerical_analysis_results:
                numerical_analysis_results_df = pd.concat(numerical_analysis_results, axis=1)
            else:
                numerical_analysis_results_df = pd.DataFrame()

            return numerical_analysis_results_df, textual_analysis_sections

        except Exception as e:
            print(f"An exception occurred: {e}")
            traceback.print_exc()
            return pd.DataFrame(), []

    def load_and_preprocess_data(self, file_path):
        """
        Loads a CSV file and preprocesses it for analysis.
        :param file_path: Path to the CSV file.
        :return: Preprocessed dataframe.
        """
        try:
            df = pd.read_csv(file_path)
            return self.normalize_data(df)
        except Exception as e:
            print(f"Error in loading and preprocessing data: {e}")
            return pd.DataFrame()

    def normalize_data(self, data_frame):
        """
        Normalizes the data in a dataframe for consistent formatting.
        :param data_frame: The dataframe to normalize.
        :return: Normalized dataframe.
        """
        try:
            for col in data_frame.columns:
                # Normalize text data
                if data_frame[col].dtype == object:
                    data_frame[col] = data_frame[col].str.lower().str.strip()

                # Attempt to convert to datetime if possible
                try:
                    data_frame[col] = pd.to_datetime(data_frame[col], errors='coerce')
                except:
                    pass

                # Handling mixed data types within a column
                if data_frame[col].map(type).nunique() > 1:
                    # Convert to string for consistent data type
                    data_frame[col] = data_frame[col].astype(str)

            return data_frame
        except Exception as e:
            print(f"Error in normalizing data: {e}")
            return data_frame

    def compare_data(self, data_frames, selected_columns):
        """
        Compares data across multiple CSV files based on selected columns.
        Performs a row-wise comparison.
        :param data_frames: Dictionary of dataframes loaded from CSV files.
        :param selected_columns: List of columns to compare.
        :return: DataFrame with comparison results.
        """
        print("Selected Columns:", selected_columns)  # Debug print

        comparison_results = pd.DataFrame()
        for column in selected_columns:
            print(f"Processing column: {column}")  # Debug print

            combined_column_data = pd.DataFrame()
            for filename, df in data_frames.items():
                if column in df.columns:
                    combined_column_data[filename] = df[column]
                    print(f"Data from {filename}:", df[column].to_list())  # Debug print

            # Check if any data was added for comparison
            if not combined_column_data.empty:
                comparison_results[column] = combined_column_data.apply(lambda x: x.nunique() == 1, axis=1)
            else:
                print(f"No data to compare for column: {column}")  # Debug print

        # If comparison_results is empty, add a message column
        if comparison_results.empty:
            comparison_results['Message'] = ["No comparable data found for the selected columns."]

        print("Comparison Results:", comparison_results)  # Debug print

        numerical_results, textual_results = self.classify_and_analyze_data(data_frames)
        self.analysisComplete.emit({'numerical_results': numerical_results, 'textual_results': textual_results})
        return numerical_results, textual_results

    def automatic_column_selection(self, data_frames):
        selected_columns = set()  # Changed to a set to avoid duplicates
        for df in data_frames.values():
            for col in df.columns:
                if df[col].notna().all():
                    selected_columns.add(col)
        return list(selected_columns)  # Convert back to a list for consistency

    def perform_data_comparison(self, data_frames, selected_columns):
        """
        Compares data across selected columns to identify correlations, commonalities, or anomalies.
        """
        comparison_results = {}
        # Logic for comparing data across the selected columns
        return comparison_results

    def execute_analysis(self, data_frames, automatic=True, selected_columns=None):
        """
        Orchestrates the execution of data analysis, either automatic or based on selected columns.
        """
        if automatic:
            selected_columns = self.automatic_column_selection(data_frames)

        # Proceed with selected columns for further analysis
        nlp_results = self.perform_basic_nlp_analysis(data_frames, selected_columns)
        comparison_results = self.perform_data_comparison(data_frames, selected_columns)

        # Combine results into a single report
        analysis_report = {
            'nlp_results': nlp_results,
            'comparison_results': comparison_results
        }
        return analysis_report

    def aggregate_data(self, data_frames, selected_columns, aggregation_type='sum'):
        """
        Aggregates data, performing operations like sum, average, etc.
        Handles missing data and non-numeric columns.
        :param data_frames: Dictionary of dataframes loaded from CSV files.
        :param selected_columns: List of columns for aggregation.
        :param aggregation_type: Type of aggregation ('sum', 'average', 'count', etc.).
        :return: DataFrame with aggregated data results.
        """
        global aggregated_value
        aggregation_results = pd.DataFrame()
        for column in selected_columns:
            combined_column_data = pd.concat([df[column] for df in data_frames.values() if column in df.columns])
            if combined_column_data.dtype.kind in 'bfc':  # Check if the data type is numeric
                if aggregation_type == 'sum':
                    aggregated_value = combined_column_data.sum()
                elif aggregation_type == 'average':
                    aggregated_value = combined_column_data.mean()
                elif aggregation_type == 'count':
                    aggregated_value = combined_column_data.count()
                # Add more aggregation types as needed
                aggregation_results.loc[column, aggregation_type] = aggregated_value
            else:
                aggregation_results.loc[column, aggregation_type] = 'Non-numeric column'
        return aggregation_results

    def identify_common_columns(self, data_frames):
        """
        Identifies columns that are common across multiple dataframes.
        :param data_frames: Dictionary of dataframes.
        :return: List of common columns.
        """
        all_columns = [set(df.columns) for df in data_frames.values()]
        common_columns = set.intersection(*all_columns)
        return list(common_columns)

    def find_data_discrepancies(self, data_frames, common_columns):
        """
        Finds discrepancies in common columns across dataframes.
        :param data_frames: Dictionary of dataframes.
        :param common_columns: List of common columns to check.
        :return: Dictionary with discrepancies information.
        """
        discrepancies = {}
        for col in common_columns:
            values = [set(df[col].dropna().unique()) for df in data_frames.values() if col in df.columns]
            discrepancies[col] = set.union(*values) - set.intersection(*values)
        return discrepancies

    def calculate_statistics(self, data_frames, common_columns):
        """
        Calculates statistical measures for numerical columns.
        :param data_frames: Dictionary of dataframes.
        :param common_columns: List of common columns to analyze.
        :return: Dictionary with statistical data.
        """
        statistics = {}
        for col in common_columns:
            col_data = pd.concat([df[col] for df in data_frames.values() if col in df.columns], ignore_index=True)
            if col_data.dtype.kind in 'bfc':  # Check if the data type is numeric
                statistics[col] = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'mode': col_data.mode().tolist()  # mode can have multiple values
                }
        return statistics

    def identify_patterns(self, data_frame, column):
        """
        Identifies patterns or anomalies in a given column of the dataframe.
        Currently, implements a simple outlier detection.
        :param data_frame: The dataframe to analyze.
        :param column: The column to analyze for patterns.
        :return: DataFrame with identified patterns, or None.
        """
        try:
            if data_frame[column].dtype.kind in 'bfc':
                mean = data_frame[column].mean()
                std = data_frame[column].std()
                return data_frame[(data_frame[column] < mean - 2 * std) | (data_frame[column] > mean + 2 * std)]
            else:
                print(f"Column '{column}' is not numeric and cannot be analyzed for patterns.")
                return None
        except Exception as e:
            print(f"Error in identifying patterns: {e}")
            return None

    def analyze_correlations(self, data_frames, columns):
        """
        Analyzes correlations between different data columns.
        :param data_frames: Dictionary of dataframes.
        :param columns: List of column pairs to analyze for correlation.
        :return: Dictionary with correlation results.
        """
        correlations = {}
        for col1, col2 in columns:
            combined_data = pd.concat([df[[col1, col2]].dropna() for df in data_frames.values()], ignore_index=True)
            corr, _ = pearsonr(combined_data[col1], combined_data[col2])
            correlations[(col1, col2)] = corr
        return correlations

    def identify_patterns_in_numeric_columns(self, data_frames):
        """
        Identifies patterns or anomalies in numeric columns of the dataframes.
        :param data_frames: Dictionary of dataframes.
        :return: Dictionary with patterns identified in each numeric column.
        """
        patterns = {}
        try:
            for filename, df in data_frames.items():
                numeric_columns = df.select_dtypes(include='number').columns
                for column in numeric_columns:
                    pattern = self.identify_patterns(df, column)
                    if pattern is not None and not pattern.empty:
                        patterns[(filename, column)] = pattern
            return patterns
        except Exception as e:
            print(f"Error in identifying patterns across dataframes: {e}")
            return {}
    def perform_regression_analysis(df, independent_var, dependent_var):
        X = df[independent_var].values.reshape(-1, 1)
        y = df[dependent_var]

        regression_model = LinearRegression()
        regression_model.fit(X, y)
        y_pred = regression_model.predict(X)

        return regression_model.coef_, regression_model.intercept_, y_pred


class NLPAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text.
        """
        return self.sia.polarity_scores(text)

    def extract_keywords(self, text, num_keywords=10):
        """
        Extracts and returns the most frequent keywords from a given text.
        """
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove punctuation and convert to lower case
        tokens = [word.lower() for word in tokens if word.isalpha()]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Get frequency distribution of the tokens
        freq_dist = FreqDist(filtered_tokens)

        # Get the most common words
        keywords = [word for word, freq in freq_dist.most_common(num_keywords)]

        return keywords

    def perform_basic_nlp_analysis(self, data_frames, columns):
        """
        Placeholder for more advanced NLP analysis.
        """
        nlp_results = {}
        # Add logic for advanced NLP tasks here
        return nlp_results

    # Potential additional methods for tokenization, POS tagging, NER, etc.
