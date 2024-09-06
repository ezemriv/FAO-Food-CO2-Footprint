import os
import re

# Define the directories for sample tables and zip files
sample_tables_dir = r'..\data\FAOSTAT\sample_data\torename'
zip_files_dir = r'G:\My Drive\NUCLIO_drive\TFM_drive\fao_carbon_footprint_gdrive\data\FAOSTAT\all_raw'

# Get a list of sample table filenames
sample_table_files = os.listdir(sample_tables_dir)

# Regular expression to match the prefix pattern and extract the relevant filename part
pattern = re.compile(r'^\d+-sample_(.*)')

# Process each sample table filename
for sample_table in sample_table_files:
    match = pattern.match(sample_table)
    if match:
        # Extract the relevant part of the filename
        relevant_part = match.group(1)
        
        # Construct the corresponding zip file name
        zip_file_name = relevant_part.replace('.csv', '.zip')
        
        # Form the full paths to the zip file (old name) and the new zip file name
        old_zip_file_path = os.path.join(zip_files_dir, zip_file_name)
        new_zip_file_path = os.path.join(zip_files_dir, sample_table.replace('.csv', '.zip'))
        
        # Rename the zip file if it exists
        if os.path.exists(old_zip_file_path):
            os.rename(old_zip_file_path, new_zip_file_path)
            print(f"Renamed {old_zip_file_path} to {new_zip_file_path}")
        else:
            print(f"Zip file {old_zip_file_path} does not exist and cannot be renamed.")