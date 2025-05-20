import pandas as pd

# Dataloading and Pre-reqs

# Load the January 2023 data
df_jan = pd.read_parquet('./hw1/data/yellow_tripdata_2023-01.parquet') # Make sure the path to your file is correct

# Ensure the datetime columns are in datetime format
df_jan['tpep_pickup_datetime'] = pd.to_datetime(df_jan['tpep_pickup_datetime'])
df_jan['tpep_dropoff_datetime'] = pd.to_datetime(df_jan['tpep_dropoff_datetime'])

# Calculate duration
df_jan['duration'] = df_jan['tpep_dropoff_datetime'] - df_jan['tpep_pickup_datetime']

# Convert duration to minutes
df_jan['duration'] = df_jan['duration'].dt.total_seconds() / 60

# Calculate the standard deviation of the duration
std_dev_duration = df_jan['duration'].std()

## Start A3

# Original number of records
original_records_count = len(df_jan)

# Filter out the outliers
df_jan_filtered = df_jan[(df_jan['duration'] >= 1) & (df_jan['duration'] <= 60)].copy() # Use .copy() to avoid SettingWithCopyWarning

# Number of records after dropping outliers
filtered_records_count = len(df_jan_filtered)

# Calculate the fraction of records left
fraction_left = filtered_records_count / original_records_count
print(f"Fraction of records left after dropping outliers: {fraction_left}")