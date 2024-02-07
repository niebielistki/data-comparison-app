import pytest
from app.data_analyzer import DataAnalyzer
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import time
from app.csv_handler import CSVHandler

@pytest.fixture
def analyzer():
    return DataAnalyzer()

@pytest.fixture
def csv_handler():
    return CSVHandler()  # Replace with the actual constructor if different

# Fixture for setting up data
@pytest.fixture
def setup_data_frames():
    # Creating sample data frames for testing
    df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [7, 8, 9]})
    df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    return {'df1.csv': df1, 'df2.csv': df2, 'df3.csv': df3}


def test_compare_data(analyzer, setup_data_frames):
    data_frames = setup_data_frames
    selected_columns = ['A', 'B']

    # Performing the comparison
    comparison_results = analyzer.compare_data(data_frames, selected_columns)

    # Creating the expected results DataFrame
    expected_results = pd.DataFrame({
        'A': [True, True, True],  # True for all rows as column 'A' is the same across all dataframes
        'B': [False, False, False]  # False for all rows as column 'B' differs across dataframes
    })

    # Asserting the comparison results match the expected results
    pd.testing.assert_frame_equal(comparison_results, expected_results)

def test_aggregate_data_sum(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    aggregated_data = analyzer.aggregate_data(data_frames, ['A', 'B'], 'sum')

    assert aggregated_data.loc['A', 'sum'] == 140  # 10+20+50+60
    assert aggregated_data.loc['B', 'sum'] == 220  # 30+40+70+80

def test_aggregate_data_average(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    aggregated_data = analyzer.aggregate_data(data_frames, ['A', 'B'], 'average')

    assert np.isclose(aggregated_data.loc['A', 'average'], 35.0)  # (10+20+50+60)/4
    assert np.isclose(aggregated_data.loc['B', 'average'], 55.0)  # (30+40+70+80)/4

def test_aggregate_data_non_numeric_column(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    aggregated_data = analyzer.aggregate_data(data_frames, ['C'], 'sum')

    assert aggregated_data.loc['C', 'sum'] == 'Non-numeric column'

def test_identify_common_columns(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    common_columns = analyzer.identify_common_columns(data_frames)

    # Asserting that common columns are correctly identified
    assert set(common_columns) == {'A', 'B', 'C'}

def test_find_data_discrepancies(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    common_columns = analyzer.identify_common_columns(data_frames)
    discrepancies = analyzer.find_data_discrepancies(data_frames, common_columns)

    # Expected discrepancies can be adjusted based on the test data setup
    expected_discrepancies = {
        'A': {10, 20, 50, 60},
        'B': {30, 40, 70, 80},
        'C': {'apple', 'banana', 'cherry', 'date'}
    }
    assert discrepancies == expected_discrepancies

def test_calculate_statistics(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    common_columns = analyzer.identify_common_columns(data_frames)
    statistics = analyzer.calculate_statistics(data_frames, common_columns)

    # Asserting that statistics are calculated correctly
    # Adjust these assertions based on your test data setup
    assert statistics['A']['mean'] == 35.0  # Mean of values in column A
    assert statistics['B']['median'] == 55.0  # Median of values in column B


def test_identify_patterns(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frame = setup_data_frames['df1']
    column = 'A'

    # Calculate mean and standard deviation for the column
    mean = data_frame[column].mean()
    std = data_frame[column].std()

    patterns = analyzer.identify_patterns(data_frame, column)

    # Calculate the expected pattern
    expected_pattern = data_frame[(data_frame[column] < mean - 2 * std) | (data_frame[column] > mean + 2 * std)]

    # Check if the patterns identified by the function match the expected pattern
    assert patterns.equals(expected_pattern)


def test_analyze_correlations(setup_data_frames):
    analyzer = DataAnalyzer()
    data_frames = setup_data_frames
    columns = [('A', 'B')]  # Adjust based on your data

    correlations = analyzer.analyze_correlations(data_frames, columns)

    # Calculate the expected correlation for comparison
    combined_data = pd.concat([df[['A', 'B']].dropna() for df in data_frames.values()], ignore_index=True)
    expected_correlation, _ = pearsonr(combined_data['A'], combined_data['B'])

    assert np.isclose(correlations[('A', 'B')], expected_correlation)

def test_load_with_invalid_path(analyzer):
    invalid_path = "/tests/data/bundle47"
    result = analyzer.load_and_preprocess_data(invalid_path)
    assert result == {} or all(df.empty for df in result.values() if isinstance(df, pd.DataFrame))

def test_performance_large_dataset(analyzer):
    some_acceptable_duration = 10  # seconds
    start_time = time.time()
    analyzer.load_and_preprocess_data("/Users/wiktoria/PycharmProjects/DataComparisonApp/data/data_4")
    end_time = time.time()
    assert end_time - start_time < some_acceptable_duration

def test_integration_with_csv_handler(analyzer, csv_handler):
    # Path to a directory containing test CSV files
    folder_path = "/data/data_4"

    # Load the CSV files using CSVHandler
    csv_handler.read_csv_files(folder_path)

    # Load and preprocess the data using DataAnalyzer
    data_frames = analyzer.load_and_preprocess_data(folder_path)

    # Define columns to aggregate and the type of aggregation
    selected_columns = ['A', 'B']  # Example columns, adjust as needed
    aggregation_type = 'sum'  # Example aggregation type

    # Perform the aggregation
    analysis_result = analyzer.aggregate_data(data_frames, selected_columns, aggregation_type)

    # Define the expected result
    # This should be based on the known expected outcome for the given test data
    # For example, if you know the sum of columns 'A' and 'B' in your test data, set it here
    expected_result = pd.DataFrame({
        'A': [sum_of_column_A],  # Replace with the actual expected sum of column 'A'
        'B': [sum_of_column_B]   # Replace with the actual expected sum of column 'B'
    })

    # Assert that the analysis result matches the expected result
    pd.testing.assert_frame_equal(analysis_result, expected_result)









