import pandas as pd

# Specify the filename
filename = 'test.csv'
# Specify the column name you want to extract
column_name = 'license_number'

try:
    # Read the CSV file using pandas with additional parameters to handle common issues
    df = pd.read_csv(filename, on_bad_lines='skip', quoting=3)
    
    # Extract the column data and convert it to a list
    column_data = df[column_name].tolist()
    
    # Remove duplicates by converting the list to a set and then back to a list
    column_data = list(set(column_data))
    
    # Select only the first four unique entries
    first_four_data = column_data[:4]
except pd.errors.ParserError as e:
    print(f'Error parsing the CSV file: {e}')
except KeyError:
    print(f'Column "{column_name}" does not exist in the CSV file.')

# Print the list to verify
print(first_four_data)
