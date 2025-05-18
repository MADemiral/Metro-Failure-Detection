import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# Load preprocessed data
df = pd.read_csv('data/dataset_train_processed.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define original anomaly window
start_original = pd.Timestamp("2020-04-11 11:50:00")
end_original = pd.Timestamp("2020-04-12 23:30:00")

# Expand 6 hours before and after
start_window = start_original - timedelta(hours=6)
end_window = end_original + timedelta(hours=6)

# Filter data in the extended window
df = df[(df['timestamp'] >= start_window) & (df['timestamp'] <= end_window)]

# Digital feature names (you can adjust if different)
digital_features = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']

# Extract digital features
X_digital = df[digital_features].copy()

# Normalize (autoencoder was trained on scaled data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digital)
X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))


from keras.models import load_model

# Load trained autoencoder
autoencoder = load_model('sae_digital.h5')

# Step 1: Scale input
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digital)  # shape: (n_samples, 8)

# Step 2: Reshape for autoencoder input: (samples, timesteps, features)
X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # shape: (n_samples, 1, 8)

# Step 3: Predict
X_reconstructed_3d = autoencoder.predict(X_scaled_3d)  # shape: (n_samples, 1, 8)

# Step 4: Flatten back to 2D
X_reconstructed = X_reconstructed_3d.reshape(X_scaled.shape)  # shape: (n_samples, 8)

# Step 5: Compute reconstruction error
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)


from sklearn.ensemble import RandomForestRegressor

# Fit model to predict reconstruction error
regressor = RandomForestRegressor(n_estimators=50, random_state=42)
regressor.fit(X_scaled, reconstruction_error)

import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_scaled)

# Plot SHAP summary
shap.summary_plot(shap_values, features=X_scaled, feature_names=digital_features, plot_type='bar')


