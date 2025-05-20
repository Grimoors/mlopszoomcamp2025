import pandas as pd

# Load the January 2023 data
df_jan = pd.read_parquet('./hw1/data/yellow_tripdata_2023-01.parquet') # Make sure the path to your file is correct

# Get the number of columns
num_columns = df_jan.shape[1]
print(f"Number of columns in January data: {num_columns}")
