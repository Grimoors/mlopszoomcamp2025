import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

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

# Start A4

# Select the relevant columns
categorical_cols = ['PULocationID', 'DOLocationID']
df_features = df_jan_filtered[categorical_cols].copy() # Use .copy()

# Cast IDs to strings
df_features['PULocationID'] = df_features['PULocationID'].astype(str)
df_features['DOLocationID'] = df_features['DOLocationID'].astype(str)

# Turn the dataframe into a list of dictionaries
records = df_features.to_dict(orient='records')

# Fit a dictionary vectorizer
dv = DictVectorizer()
feature_matrix = dv.fit_transform(records)

# Start A5

# Prepare the target variable (duration from the filtered dataframe)
y_train = df_jan_filtered['duration'].values

# Train a plain linear regression model
lr = LinearRegression()
lr.fit(feature_matrix, y_train)

# Make predictions on the training data
y_pred_train = lr.predict(feature_matrix)

# Calculate the RMSE on the training data
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"RMSE on train: {rmse_train}")