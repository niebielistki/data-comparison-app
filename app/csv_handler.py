"""
csv_handler.py

This module contains the CSVHandler class, which is designed to facilitate the loading, analysis, and preprocessing of CSV files within the DataComparisonApp. It offers functionality to read multiple CSV files from a specified directory, identify common columns across these files, and perform preliminary data analysis tasks such as detecting outliers, calculating missing data percentages, and identifying duplicates. Additionally, it supports more advanced analysis by suggesting comparison scenarios based on the data's structure and content, validating column contents against specified criteria (e.g., email or numeric validation), and generating comparison possibilities for effective data analysis. The CSVHandler serves as a critical component of the DataComparisonApp by providing the foundational data manipulation capabilities required for further analysis and visualization.

Key functionalities include:
- Reading CSV files and storing their contents as pandas DataFrames.
- Identifying common columns across multiple CSV files to suggest potential comparison points.
- Mapping columns to files for easy tracking of data sources.
- Validating column contents based on predefined criteria (e.g., email, numeric).
- Analyzing comparison scenarios to guide the user in selecting meaningful data points for analysis.
- Detecting outliers, calculating missing data percentages, identifying duplicates, and checking for invalid entries within the data for quality control.
- Generating suggestions for data comparisons based on the analysis of common columns and data types.

Designed to support a variety of data analysis workflows within the app, the CSVHandler class provides a robust toolkit for the initial handling and inspection of CSV data, thereby enabling users to make informed decisions about which data columns to compare and analyze.
"""

import os
import pandas as pd
import numpy as np


class CSVHandler:

    def read_csv_files(self, folder_path):
        data_frames = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        print(f"Warning: The file '{filename}' is empty.")
                    else:
                        data_frames[filename] = df
                except pd.errors.EmptyDataError:
                    print(f"Warning: No columns to parse from file '{filename}'.")
                except Exception as e:
                    print(f"Error reading '{filename}': {e}")
        return data_frames

    def identify_common_columns(self, data_frames):
        column_count = {}
        for df in data_frames.values():
            for column in df.columns:
                column_count[column] = column_count.get(column, 0) + 1
        common_columns = [column for column, count in column_count.items() if count >= 2]
        return common_columns

    def files_containing_column(self, column, data_frames):
        files_with_column = []
        for file_path, df in data_frames.items():
            if column in df.columns:
                files_with_column.append(os.path.basename(file_path))
        return files_with_column

    def map_columns_to_files(self, data_frames):
        column_file_map = {}
        for filename, df in data_frames.items():
            for column in df.columns:
                if column in column_file_map:
                    column_file_map[column].add(filename)
                else:
                    column_file_map[column] = {filename}
        return column_file_map

    def validate_column_content(self, data_frames, column_name, validation_type):
        validation_results = {}
        for filename, df in data_frames.items():
            if column_name in df.columns:
                if validation_type == 'email':
                    valid = df[column_name].str.contains("^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", regex=True)
                elif validation_type == 'numeric':
                    valid = df[column_name].apply(lambda x: isinstance(x, (int, float)))
                else:
                    valid = pd.Series([True] * len(df))  # Default to valid if no type specified
                validation_results[filename] = valid
        return validation_results

    def analyze_comparison_scenarios(self, column_file_map):
        comparison_scenarios = {
            'no_common_columns': [],
            'with_common_columns': {}
        }

        for column, files in column_file_map.items():
            if len(files) > 1:
                comparison_scenarios['with_common_columns'].setdefault(tuple(sorted(files)), []).append(column)
            else:
                comparison_scenarios['no_common_columns'].extend(files)

        return comparison_scenarios

    def analyze_data_types(self, data_frames):
        data_type_report = {}
        for filename, df in data_frames.items():
            data_type_report[filename] = df.dtypes.apply(lambda x: str(x)).to_dict()
        return data_type_report

    def generate_comparison_possibilities(self, comparison_scenarios):
        comparison_possibilities = {}
        for file_set, columns in comparison_scenarios['with_common_columns'].items():
            if len(columns) >= 1:  # Assuming you want to include even single-column matches
                comparison_possibilities[file_set] = columns

        # Sort the possibilities based on the number of common columns, descending
        sorted_possibilities = sorted(comparison_possibilities.items(), key=lambda item: len(item[1]), reverse=True)
        return dict(sorted_possibilities)

    def suggest_comparisons(self, comparison_possibilities):
        suggestions = {}
        for file_set, columns in comparison_possibilities.items():
            if len(columns) > 1:  # More meaningful if multiple columns in common
                suggestion = f"Compare {', '.join(columns)} between {', '.join(file_set)}"
                suggestions[suggestion] = (file_set, columns)
        return suggestions

    def detect_outliers(self, df):
        outlier_info = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
            outlier_info[column] = len(outliers)
        return outlier_info

    def calculate_missing_data_percentage(self, df):
        return df.isnull().mean() * 100

    def identify_duplicates(self, df):
        return df.duplicated().sum()

    def check_invalid_entries(self, df, column_name, valid_range=None):
        if valid_range:
            invalid_entries = df[(df[column_name] < valid_range[0]) | (df[column_name] > valid_range[1])]
        else:
            invalid_entries = pd.DataFrame()
        return len(invalid_entries)
