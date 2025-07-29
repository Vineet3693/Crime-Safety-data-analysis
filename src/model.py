
# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import joblib to save the model

# Load the processed data
data = pd.read_csv('../data/processed_data.csv')

# Define features and target variable
X = data.drop(['id', 'date', 'time', 'victim_age'], axis=1)
y = data['victim_age']  # Example target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(model, './model.pkl')  # Save the model in the src directory
