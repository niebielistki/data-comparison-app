#! python3
# data_analyzer.py

import pandas as pd
from csv_handler import CSVHandler
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.stats import linregress
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
import random


class AnalysisUtilities:
    @staticmethod
    def detect_trend(data_series):
        """
        Detects the trend in a data series using linear regression.

        :param data_series: Pandas Series, a series of data points.
        :return: String indicating the trend ('increasing', 'decreasing', or 'stable').
        """
        x = range(len(data_series))
        slope, _, _, _, _ = linregress(x, data_series)
        if slope > 0:
            return "increasing"
        elif slope < 0:
            return "decreasing"
        else:
            return "stable"

    @staticmethod
    def calculate_volatility(data_series, window=30):
        """
        Calculates the volatility of a data series over a specified window.

        :param data_series: Pandas Series, a series of data points.
        :param window: Integer, the size of the rolling window to calculate volatility.
        :return: Pandas Series, the volatility of the data series.
        """
        return data_series.rolling(window=window).std()

    @staticmethod
    def calculate_mean(data_series):
        """
        Calculates the mean of a data series.

        :param data_series: Pandas Series, a series of data points.
        :return: Float, the mean of the data series.
        """
        return data_series.mean()

    @staticmethod
    def calculate_median(data_series):
        """
        Calculates the median of a data series.

        :param data_series: Pandas Series, a series of data points.
        :return: Float, the median of the data series.
        """
        return data_series.median()

    @staticmethod
    def calculate_mode(data_series):
        """
        Calculates the mode of a data series.

        :param data_series: Pandas Series, a series of data points.
        :return: Pandas Series, the mode(s) of the data series.
        """
        return data_series.mode()

    @staticmethod
    def calculate_variance(data_series):
        """
        Calculates the variance of a data series.

        :param data_series: Pandas Series, a series of data points.
        :return: Float, the variance of the data series.
        """
        return data_series.var()

    @staticmethod
    def calculate_std_dev(data_series):
        """
        Calculates the standard deviation of a data series.

        :param data_series: Pandas Series, a series of data points.
        :return: Float, the standard deviation of the data series.
        """
        return data_series.std()

    @staticmethod
    def assess_missing_data(data_frame):
        """
        Assesses the missing data in a DataFrame.

        :param data_frame: Pandas DataFrame, the DataFrame to assess for missing data.
        :return: Two Pandas Series, the count and percentage of missing data for columns with missing values.
        """
        missing_data = data_frame.isnull().sum()
        total_data = len(data_frame)
        missing_percentage = (missing_data / total_data) * 100
        return missing_data[missing_data > 0].sort_values(ascending=False), missing_percentage[
            missing_percentage > 0].sort_values(ascending=False)

    @staticmethod
    def detect_outliers(data_series):
        """
        Detects outliers in a data series using the Interquartile Range (IQR) method.

        :param data_series: Pandas Series, a series of data points.
        :return: Pandas Series, the outliers in the data series.
        """
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data_series[(data_series < lower_bound) | (data_series > upper_bound)]

    @staticmethod
    def select_random_year(df):
        years = df.index.year.unique().tolist()
        random_year = random.choice(years)
        return random_year

    @staticmethod
    def higher_or_lower(df, year, variable):
        previous_years_avg = df[df.index.year < year][variable].mean()
        current_year_value = df[df.index.year == year][variable].mean()

        return "higher" if current_year_value > previous_years_avg else "lower"

    @staticmethod
    def get_volatility_years(df, volatility_column='volatility_score'):
        """
        Identifies the years with the highest and lowest volatility.

        :param df: A DataFrame with an index set to years and a column for volatility scores.
        :param volatility_column: The name of the column containing volatility scores.
        :return: A tuple containing the years with the most and least volatility.
        """
        if volatility_column not in df.columns:
            raise ValueError(f"Column '{volatility_column}' not found in DataFrame.")

        most_volatile_year = df[volatility_column].idxmax()
        least_volatile_year = df[volatility_column].idxmin()
        return (most_volatile_year, least_volatile_year)

    @staticmethod
    def calculate_trend_strength(data_series):
        """
        Quantifies the strength of a trend based on the slope of a linear regression line.

        :param data_series: Pandas Series, a series of data points.
        :return: Float, the absolute value of the slope, indicating trend strength.
        """
        x = np.arange(len(data_series))
        y = data_series.values
        slope, _, _, _, _ = linregress(x, y)
        return abs(slope)

    @staticmethod
    def calculate_volatility_score(data_series):
        """
        Provides a normalized score of volatility based on the standard deviation relative to the mean.

        :param data_series: Pandas Series, a series of data points.
        :return: Float, the volatility score.
        """
        if data_series.mean() != 0:  # Avoid division by zero
            return data_series.std() / data_series.mean()
        return 0

    @staticmethod
    def identify_anomalies(data_series):
        """
        Identifies anomalies in a data series using the Z-score method.

        :param data_series: Pandas Series, a series of data points.
        :return: A DataFrame with outliers and their z-scores.
        """
        z_scores = zscore(data_series)
        anomalies = data_series[(z_scores < -3) | (z_scores > 3)]
        return anomalies.to_frame(name='Value').assign(Z_Score=z_scores[(z_scores < -3) | (z_scores > 3)])

    @staticmethod
    def summarize_data_characteristics(data_series):
        """
        Aggregates analyses into a summary of data's characteristics.

        :param data_series: Pandas Series, a series of data points.
        :return: Dictionary summarizing the data's trend strength, volatility score, and anomalies.
        """
        trend_strength = AnalysisUtilities.calculate_trend_strength(data_series)
        volatility_score = AnalysisUtilities.calculate_volatility_score(data_series)
        anomalies = AnalysisUtilities.identify_anomalies(data_series)

        summary = {
            'Trend Strength': trend_strength,
            'Volatility Score': volatility_score,
            'Number of Anomalies': len(anomalies),
            'Anomalies Detail': anomalies
        }
        return summary

    @staticmethod
    def calculate_percentage_change(data_series):
        """Calculates the percentage change between the first and last value of a Pandas Series."""
        if len(data_series) < 2 or data_series.iloc[0] == 0:
            return None  # Can't calculate change from a single data point or from zero
        return ((data_series.iloc[-1] - data_series.iloc[0]) / data_series.iloc[0]) * 100

class DataAnalyzer(QObject):
    analysisComplete = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.csv_handler = CSVHandler()


    # Templates for Textual Analysis
    def get_analysis_templates(self, analysis_type, data_type, tags=None):
        """
        Returns the appropriate analysis templates based on the analysis and data type,
        including tags for dynamic selection based on data characteristics.
        :param analysis_type: A string indicating the type of analysis ('yearly', 'monthly', 'daily', 'data_cleaning', 'descriptive_stats').
        :param data_type: A string indicating the data type ('numeric', 'text', 'both').
        :return: A list of templates as strings.
        """
        templates = {
            'yearly': {
                'numeric': {
                    'tags': {
                        'strong_increase': [
                            "The year {year1} to {year2} showed a remarkable increase in {variable}, with a growth of {percentage}%, indicating strong upward momentum.",
                            "Between {year1} and {year2}, {variable} experienced a robust increase, marking the period with the most significant growth rate of {percentage}%."
                        ],
                        'significant_volatility': [
                            "Throughout the period from {start_year} to {end_year}, {variable} demonstrated significant volatility, especially in {most_volatile_year}.",
                            "The data from {start_year} to {end_year} reveals pronounced fluctuations in {variable}, with {most_volatile_year} witnessing the highest volatility."
                        ],
                        'anomaly': [
                            "An unusual observation in {year}, where {variable} was notably {higher_or_lower} than expected, marking an anomaly.",
                            "The year {year} presented an unexpected deviation in {variable}, diverging sharply from the trend and signaling a potential anomaly."
                        ],
                        'general': [
                            "A comparison from {start_year} to {end_year} shows {variable} fluctuating, with a net change of {value}.",
                            "Year-over-year, {variable} demonstrated a {trend} trend.",
                            "The year {year1} marked a significant {trend} in {variable}, deviating from previous trends.",
                            "The cumulative change in {variable} over the last {number} years was {value}, indicating a long-term {trend}.",
                            "An alternating pattern of {trend} in {variable} was noted every alternate year from {start_year} to {end_year}.",
                            "Over the span from {start_year} to {end_year}, {variable} exhibited a {trend} trend, with a notable shift in {year} that deviated from earlier patterns."
                        ]
                    }
                },
                'text': ["In {year}, the {column_name} showed a notable {trend} in {variable}.",
                         "The {column_name} experienced its peak in {year}, with significant {metrics}.",
                         "During {year}, a shift towards {column_name} was observed, influencing {variable} considerably.",
                         "In {year}, {column_name} marked a turning point, with a marked {trend} in {variable}.",
                         "A consistent trend in {column_name} across consecutive {year} indicates a stable market for {variable}.",
                         "Seasonal variations are evident in {column_name}, with {year} typically showing a distinct pattern in {variable}.",
                         "Comparative analysis reveals that {year} was pivotal for {column_name}, diverging from previous trends in {variable}.",
                         "Anomaly detection in {year} for {column_name} suggests an unusual pattern in {variable}, warranting further investigation."
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

        selected_category = templates.get(analysis_type, {})
        selected_type = selected_category.get(data_type, {})
        if tags:
            filtered_templates = []
            for tag in tags:
                # Navigate through the nested structure to access templates by tags
                tag_templates = templates.get(analysis_type, {}).get(data_type, {}).get('tags', {}).get(tag, [])
                filtered_templates.extend(tag_templates)
            return filtered_templates

            # Fallback to return all templates if no tags are provided or found
        return templates.get(analysis_type, {}).get(data_type, [])

    def determine_template_tags(self, analysis_data):
        """
        Determines appropriate template tags based on analysis data characteristics.
        """
        tags = []
        # Example conditionals to determine tags
        if analysis_data['trend_strength'] > some_threshold:
            tags.append('strong_increase')
        if analysis_data['volatility_score'] > another_threshold:
            tags.append('significant_volatility')
        if analysis_data['anomalies_count'] > 0:
            tags.append('anomaly')
        if not tags:  # Default to general if no specific conditions are met
            tags.append('general')
        return tags

    def select_templates_based_on_tags(self, analysis_type, data_type, tags):
        """
        Dynamically selects sentence templates based on provided tags.
        """
        # Directly call get_analysis_templates with tags for filtering
        return self.get_analysis_templates(analysis_type, data_type, tags)

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


    # Integrate with AnalysisUtilities class
    def analyze_column_trend(self, column_data):
        trend = AnalysisUtilities.detect_trend(column_data)
        # Further logic to handle the trend result, possibly formatting it into a sentence for output

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
            time_columns = self.detect_time_columns({file_name: df})

            # Priority is given to numerical columns for analysis
            analysis_variable = next((col for col in df.columns if df[col].dtype in [np.float64, np.int64]), None)

            # Perform analysis based on priority
            for time_type in time_priority:
                column_name, _ = time_columns.get(time_type, (None, None))
                if column_name and analysis_variable:
                    analysis_result = {"title": f"{time_type} Analysis:", "content": ""}
                    print(f"Time type: {time_type}, Column name: {column_name}, Analysis variable: {analysis_variable}")

                    try:
                        # Call the analysis method and store the result in 'content'
                        if time_type == 'Year':
                            analysis_result["content"] = self._analyze_yearly_data(df, column_name, analysis_variable)
                        elif time_type == 'Month':
                            analysis_result["content"] = self._analyze_monthly_data(df, column_name, analysis_variable)
                        elif time_type == 'Date':
                            analysis_result["content"] = self._analyze_daily_data(df, column_name, analysis_variable)
                    except Exception as e:
                        analysis_result["content"] = f"Error analyzing {time_type.lower()} data: {e}"

                    analysis_results.append(analysis_result)

        return analysis_results

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
                    col_lower = col.lower()
                    if 'year' in col_lower:
                        time_columns['Year'][col] += 1
                    elif 'month' in col_lower:
                        time_columns['Month'][col] += 1
                    elif 'date' in col_lower:
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

    def _analyze_yearly_data(self, df, year_column, analysis_variable):
        """
        Analyzes yearly data for a given variable and generates textual analysis based on predefined templates,
        incorporating dynamic selection based on data characteristics.

        :param df: DataFrame containing the data.
        :param year_column: Column name with year data, expected to be in 'YYYY' format.
        :return: Textual analysis for yearly data.
        """
        analysis_text = []

        # Convert year column to datetime and set as index
        df[year_column] = pd.to_datetime(df[year_column], format='%Y', errors='coerce')
        if df[year_column].isnull().any():
            return "Year column contains invalid data. Cannot perform analysis."
        df.set_index(year_column, inplace=True)
        df_sorted = df.sort_values(by=year_column)

        for variable in df.columns:
            if variable == year_column:
                continue

            if df.empty or variable not in df:
                continue

            try:
                trend = AnalysisUtilities.detect_trend(df_sorted[variable])
                trend_strength = AnalysisUtilities.calculate_trend_strength(df_sorted[variable])

                # Direct calculation of volatility score for yearly data
                volatility_score = df_sorted[variable].std() / df_sorted[variable].mean()
                df_sorted['volatility_score'] = volatility_score  # This sets a constant value for all rows

                anomalies = AnalysisUtilities.identify_anomalies(df_sorted[variable])

                # Ensure there's a valid 'volatility_score' column available for analysis.
                if 'volatility_score' not in df_sorted.columns or df_sorted['volatility_score'].isna().all():
                    raise ValueError("Volatility score could not be calculated or is missing.")

                most_volatile_year, least_volatile_year = AnalysisUtilities.get_volatility_years(df_sorted, 'volatility_score')

                random_year = AnalysisUtilities.select_random_year(df)
                higher_or_lower_result = AnalysisUtilities.higher_or_lower(df, random_year, variable)
                net_change = round(df_sorted[variable].iloc[-1] - df_sorted[variable].iloc[0], 2)
                percentage_change = AnalysisUtilities.calculate_percentage_change(df_sorted[variable])
                number_of_years = df.index.max().year - df.index.min().year

                # Prepare data for analysis template
                analysis_data = {
                    'start_year': df.index.min().year,
                    'end_year': df.index.max().year,
                    'variable': variable,
                    'trend': trend,
                    'trend_strength': trend_strength,
                    'volatility_score': volatility_score,
                    'anomalies_count': len(anomalies),
                    'most_volatile_year': most_volatile_year,
                    'least_volatile_year': least_volatile_year,
                    'value': net_change,
                    'year1': df.index.min().year,
                    'year2': df.index.max().year,
                    'percentage': percentage_change,
                    'number': number_of_years,
                    'year': random_year,
                    'higher_or_lower': higher_or_lower_result
                }

                # Dynamically select and fill in the appropriate analysis templates
                template_tags = self.determine_template_tags(analysis_data)  # Determine which tags apply
                analysis_templates = self.select_templates_based_on_tags('yearly', 'numeric',
                                                                         template_tags)  # Retrieve filtered templates

                analysis_result = [template.format(**analysis_data) for template in analysis_templates]

                final_analysis = "<br>".join(analysis_result)
                analysis_text.append(final_analysis)

            except Exception as e:
                analysis_text.append(f"Error in analyzing yearly data for '{variable}': {str(e)}")
                print(traceback.format_exc())

        return "\n".join(analysis_text)

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
                # Data Cleaning Tool
                if not self.section_already_added('Data Cleaning Tools', textual_analysis_sections):
                    data_cleaning_text = self.perform_data_cleaning_analysis(df)
                    if data_cleaning_text:
                        textual_analysis_sections.append(
                            {'title': 'Data Cleaning Tools', 'content': data_cleaning_text})

                # Descriptive Statistics
                if not self.section_already_added('Descriptive Statistics', textual_analysis_sections):
                    descriptive_stats_text = self.calculate_descriptive_statistics(df).to_string()
                    if descriptive_stats_text:
                        textual_analysis_sections.append(
                            {'title': 'Descriptive Statistics', 'content': descriptive_stats_text})

                # Time Series Analysis
                if not self.section_already_added('Time Series Analysis', textual_analysis_sections):
                    time_series_analysis_text = self.analyze_time_series({file_path: df})
                    if time_series_analysis_text:
                        textual_analysis_sections.append(
                            {'title': 'Time Series Analysis', 'content': time_series_analysis_text})

                # Process each column for numerical analysis
                for column in df.columns:
                    if df[column].dtype in [np.int64, np.float64]:
                        print(f"Processing numerical column: {column}")
                        numerical_df = df[[column]]
                        stats_df = self.calculate_descriptive_statistics(numerical_df)
                        if not stats_df.empty:
                            numerical_analysis_results.append(stats_df)

                # Additional detailed textual analysis
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


    # Helper function for classify_and_analyze_data method
    def section_already_added(self, section_title, sections_list):
        """
        Checks if a section with the given title has already been added to the list.

        Args:
        - section_title (str): The title of the section to check.
        - sections_list (list): The list of sections already added.

        Returns:
        - bool: True if the section is already added, False otherwise.
        """
        return any(section['title'] == section_title for section in sections_list)

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









