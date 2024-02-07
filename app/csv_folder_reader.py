# csv_folder_reader.py
# Description: This program reads all CSV files in a specified folder and prints their contents in a readable CSV format.
# Usage: Useful for quickly viewing the contents of multiple CSV files during development or data analysis.
# Additional Feature: Indicates when a file is empty.

import os
import csv

# Define the folder path containing the CSV files
folder_path = ""  # Replace with your folder path

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        # Print the file name
        print(f"{filename}")
        # Open and read the CSV file
        with open(os.path.join(folder_path, filename), mode='r', encoding='utf-8') as file:
            # Use csv.reader to read the file
            csv_reader = csv.reader(file, delimiter=';')
            # Initialize a flag to check if the file is empty
            is_empty = True
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Check if the row is not just empty or whitespace
                if any(field.strip() for field in row):
                    print(';'.join(row))
                    is_empty = False
            # If the file is empty, print "This file is empty."
            if is_empty:
                print("(This file is empty)")
        print("\n")  # Print a new line for separation between files
