from helper_code import *
import random
import shutil
import os
from pathlib import Path

print(os.path.abspath(os.getcwd()))
print(os.path.dirname(os.path.abspath(__file__)))

data_folder = "/home/bartek/PHYSIONET-2025/training_data/all_files"
test_data_folder = "/home/bartek/PHYSIONET-2025/test_files/"
print('Finding the Challenge data...')
records = find_records(data_folder)
num_records = len(records)
print(num_records)


sami_total = []
code_15_positive = []
code_15_negative = []
other_positive = []
other_negative = []


for i in range(num_records):
    print(f"{i+1}/{num_records}")
    recording_file = os.path.join(data_folder, records[i]) 
    header_file = os.path.join(data_folder, get_header_file(records[i]))
    header = load_header(recording_file)
    recording_file = os.path.join(data_folder, records[i])  + ".dat"
    try:
        current_label= get_label(header)
    except Exception as e:
        print("Failed to load label, assigning 0")
        current_label = 0


    splitted = header.split("\n")
    source_info = [x for x in splitted if "Source" in x]
    if len(source_info) > 0:
        if "SaMi-Trop" in source_info[0]:
            sami_total.append((recording_file, header_file))
        elif "CODE" in source_info[0]:
            if current_label==0:
                code_15_negative.append((recording_file, header_file))
            else:
                code_15_positive.append((recording_file, header_file))
        else:
            if current_label==0:
                other_negative.append((recording_file, header_file))
            else:
                other_positive.append((recording_file, header_file))
    else:
        if current_label==0:
            other_negative.append((recording_file, header_file))
        else:
            other_positive.append((recording_file, header_file))


# Create test directory if it doesn't exist
test_dir = Path(test_data_folder)
test_dir.mkdir(parents=True, exist_ok=True)

def move_sample(fraction, list_files):
    if len(list_files) == 0:
        print("Empty list")
        return 

    sample_size = max(1, int(fraction * len(list_files)))  # Ensure at least 1 file if list is small
    sample_files = random.sample(list_files, sample_size)

    # Move selected files to the test directory
    for files in sample_files:
        for file in files:
            source = Path(file)
            destination = test_dir / source.name
            shutil.move(str(source), str(destination))

move_sample(0.3, sami_total)
move_sample(0.3, code_15_positive)
move_sample(0.3, code_15_negative)
move_sample(0.3, other_positive)
move_sample(0.3, other_negative)
