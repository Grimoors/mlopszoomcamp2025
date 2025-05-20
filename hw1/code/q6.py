import pandas as pd
from sklearn.feature_extraction import DictVectorizer # Already imported
from sklearn.linear_model import LinearRegression    # Already imported
from sklearn.metrics import mean_squared_error       # Already imported
import numpy as np                                  # Already imported

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

#Start A6

# --- Preprocessing February Data ---
# 1. Load February Data
df_feb = pd.read_parquet('./hw1/data/yellow_tripdata_2023-02.parquet') # Make sure path is correct

# 2. Compute Duration
df_feb['tpep_pickup_datetime'] = pd.to_datetime(df_feb['tpep_pickup_datetime'])
df_feb['tpep_dropoff_datetime'] = pd.to_datetime(df_feb['tpep_dropoff_datetime'])
df_feb['duration'] = df_feb['tpep_dropoff_datetime'] - df_feb['tpep_pickup_datetime']
df_feb['duration'] = df_feb['duration'].dt.total_seconds() / 60

# 3. Drop Outliers
df_feb_filtered = df_feb[(df_feb['duration'] >= 1) & (df_feb['duration'] <= 60)].copy()

# 4. One-Hot Encode (using the DV fitted on January data)
#    Make sure 'dv' is the DictVectorizer you fitted in Q4
categorical_cols_feb = ['PULocationID', 'DOLocationID']
df_features_feb = df_feb_filtered[categorical_cols_feb].copy()

df_features_feb['PULocationID'] = df_features_feb['PULocationID'].astype(str)
df_features_feb['DOLocationID'] = df_features_feb['DOLocationID'].astype(str)

records_feb = df_features_feb.to_dict(orient='records')
X_val = dv.transform(records_feb) # Use the dv from Q4 (fitted on Jan data)

# 5. Prepare Target Variable for validation
y_val = df_feb_filtered['duration'].values

# --- Evaluating the model ---
# Make predictions on the validation data (February)
# Make sure 'lr' is the LinearRegression model you trained in Q5
y_pred_val = lr.predict(X_val)

# Calculate the RMSE on the validation data
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"RMSE on validation: {rmse_val}")