# Import the CSVHandler class from your module
from csv_handler import CSVHandler

# Create an instance of the CSVHandler class
csv_handler = CSVHandler()

# Define the folder path containing your CSV files
folder_path = "/test_data/data_4"


def test_read_csv_files():
    data_frames = csv_handler.read_csv_files(folder_path)
    print("Test Read CSV Files: " + ("Passed" if data_frames else "Failed - No CSV files found."))
    return data_frames


def test_identify_common_columns(data_frames):
    if not data_frames:
        print("Test Identify Common Columns: Skipped - No CSV files read.")
        return None
    common_columns = csv_handler.identify_common_columns(data_frames)
    print("Test Identify Common Columns: " + ("Passed" if common_columns else "Failed - No common columns found."))
    return common_columns


def test_map_columns_to_files(data_frames):
    if not data_frames:
        print("Test Map Columns to Files: Skipped - No CSV files read.")
        return None
    column_file_map = csv_handler.map_columns_to_files(data_frames)
    print("Test Map Columns to Files: " + ("Passed" if column_file_map else "Failed - Mapping not successful."))
    return column_file_map


def test_validate_column_content(data_frames, column_name, validation_type):
    if not data_frames:
        print("Test Validate Column Content: Skipped - No CSV files read.")
        return None
    validation_results = csv_handler.validate_column_content(data_frames, column_name, validation_type)
    print(f"Test Validate Column Content for '{column_name}' ({validation_type}): " + (
        "Passed" if validation_results else "Failed - Validation not successful."))
    return validation_results


def test_analyze_comparison_scenarios(column_file_map):
    if not column_file_map:
        print("Test Analyze Comparison Scenarios: Skipped - No column file map available.")
        return None
    comparison_scenarios = csv_handler.analyze_comparison_scenarios(column_file_map)
    print("Test Analyze Comparison Scenarios: " + (
        "Passed" if comparison_scenarios else "Failed - No scenarios identified."))
    return comparison_scenarios


def test_analyze_data_types(data_frames):
    if not data_frames:
        print("Test Analyze Data Types: Skipped - No CSV files read.")
        return None
    data_type_report = csv_handler.analyze_data_types(data_frames)
    print("Test Analyze Data Types: " + ("Passed" if data_type_report else "Failed - No data type report generated."))
    return data_type_report


def test_generate_comparison_possibilities(comparison_scenarios):
    if not comparison_scenarios:
        print("Test Generate Comparison Possibilities: Skipped - No comparison scenarios available.")
        return None
    comparison_possibilities = csv_handler.generate_comparison_possibilities(comparison_scenarios)
    print("Test Generate Comparison Possibilities: " + (
        "Passed" if comparison_possibilities else "Failed - No comparison possibilities generated."))
    return comparison_possibilities

def test_suggest_comparisons(comparison_possibilities):
    if not comparison_possibilities:
        print("Test Suggest Comparisons: Skipped - No comparison possibilities available.")
        return None
    suggestions = csv_handler.suggest_comparisons(comparison_possibilities)
    print("Test Suggest Comparisons: " + ("Passed" if suggestions else "Failed - No suggestions generated."))
    return suggestions


def main():
    print("Starting CSVHandler Tests...\n")

    # Step 1: Test reading CSV files
    data_frames = test_read_csv_files()

    # Step 2: Test identifying common columns, mapping columns to files, and validating column content
    common_columns = test_identify_common_columns(data_frames)
    column_file_map = test_map_columns_to_files(data_frames)
    validation_results_email = test_validate_column_content(data_frames, 'Email', 'email')

    # Step 3: Test analyzing comparison scenarios and data types
    comparison_scenarios = test_analyze_comparison_scenarios(column_file_map)
    data_type_report = test_analyze_data_types(data_frames)

    # Step 4: Test generating comparison possibilities
    comparison_possibilities = test_generate_comparison_possibilities(comparison_scenarios)

    # Test suggesting comparisons
    comparison_possibilities = test_generate_comparison_possibilities(comparison_scenarios)
    suggestions = test_suggest_comparisons(comparison_possibilities)

    # Detailed Results
    if data_frames:
        print("\nFeatures of the Program:")
        print("- Successfully read the following CSV files:")
        for filename in data_frames.keys():
            print(f"  - {filename}")

        if common_columns:
            print("\n- Common Columns: ", common_columns)

        if column_file_map:
            print("\n- File to Column Mapping:")
            for column, files in column_file_map.items():
                print(f"  - {column}: {files}")

        if validation_results_email:
            print("\n- Column Content Validation Results for 'Email':")
            for file, result in validation_results_email.items():
                print(f"  - {file}: {result}")

        if comparison_scenarios:
            print("\n- Column Comparison Scenarios:")
            print(comparison_scenarios)

        if data_type_report:
            print("\n- Data Type Analysis Report:")
            for file, types in data_type_report.items():
                print(f"  - {file}: {types}")

        if comparison_possibilities:
            print("\n- Comparison Possibilities:")
            for files, columns in comparison_possibilities.items():
                print(f"  - {files}: {columns}")

        if suggestions:
            print("\n- Suggested Comparisons:")
            for suggestion, details in suggestions.items():
                print(f"  - {suggestion}")

    print("\nCSVHandler Tests Completed.")


if __name__ == "__main__":
    main()
