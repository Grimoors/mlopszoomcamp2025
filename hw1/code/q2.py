import pandas as pd

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
print(f"Standard deviation of trip duration in January: {std_dev_duration}")