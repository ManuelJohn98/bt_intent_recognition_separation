import os

# set root directory for this project
root_directory = os.path.dirname(os.path.abspath(__file__))
# set data directory
data_directory = os.path.join(root_directory, "data")
# set raw data directory
raw_data_directory = os.path.join(data_directory, "raw_data")
# set statistics directory
statistics_directory = os.path.join(root_directory, "statistics")
# TODO: set model directory

# statistics flag
STAT = False
