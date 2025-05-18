import pandas as pd
import numpy as np
from datetime import timedelta

# Load data
df = pd.read_csv('data/dataset_train_processed.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define original anomaly window
start_original = pd.Timestamp("2020-04-11 11:50:00")
end_original = pd.Timestamp("2020-04-12 23:30:00")

# Expand 6 hours before and after
start_window = start_original - timedelta(hours=6)
end_window = end_original + timedelta(hours=6)

# Filter data in the extended window
shap_df = df[(df['timestamp'] >= start_window) & (df['timestamp'] <= end_window)]

# Select analog features
analog_features = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Motor_current', 'Oil_temperature']
X = shap_df[analog_features]

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load model & create surrogate
import pickle
from sklearn.ensemble import RandomForestClassifier

with open('models/ocsvm_analog.pkl', 'rb') as f:
    ocsvm = pickle.load(f)

# Get anomaly labels (for surrogate)
pseudo_labels = (ocsvm.predict(X_scaled) == -1).astype(int)

# Train RandomForestClassifier as SHAP explainer
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_scaled, pseudo_labels)

# Compute SHAP
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_scaled)

# Plot
shap.summary_plot(shap_values, X_scaled, feature_names=analog_features)


