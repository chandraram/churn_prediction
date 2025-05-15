import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import os

base_dir = os.path.dirname(__file__)  # Current file's directory
file_path = os.path.join(base_dir, "data", "churn_data.csv")
df = pd.read_csv(file_path)


# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical and target
encoder = LabelEncoder()
df['Contract'] = encoder.fit_transform(df['Contract'])
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model, scaler, encoder
joblib.dump(model, 'model/churn_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(encoder, 'model/encoder.pkl')

print("Training complete. Model and transformers saved.")
