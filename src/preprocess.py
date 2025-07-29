
# preprocess.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv('../data/raw_data.csv')

# Handle missing values (example: drop rows with missing values)
data.dropna(inplace=True)

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[['crime_type', 'city', 'state', 'location_description', 'victim_gender', 'victim_race']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Combine with original data
processed_data = pd.concat([data[['id', 'date', 'time', 'victim_age']], encoded_df], axis=1)

# Save the processed data
processed_data.to_csv('../data/processed_data.csv', index=False)
