
# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the processed data
data = pd.read_csv('../data/processed_data.csv')  # Adjust the path as necessary

# Define features and target variable
X = data.drop(['id', 'date', 'time', 'victim_age'], axis=1)  # Features
y = data['victim_age']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')  # Print accuracy

# Save the trained model to a file
joblib.dump(model, './model.pkl')  # Save the model in the current directory
