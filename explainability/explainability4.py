import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
from datetime import timedelta
from tensorflow.keras.losses import MeanSquaredError

# Step 1: Load data
df = pd.read_csv('data/dataset_train_processed.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

start_original = pd.Timestamp("2020-04-11 11:50:00")
end_original = pd.Timestamp("2020-04-12 23:30:00")

# Expand 6 hours before and after
start_window = start_original - timedelta(hours=6)
end_window = end_original + timedelta(hours=6)

df = df[(df['timestamp'] >= start_window) & (df['timestamp'] <= end_window)]

# Step 2: Select digital features
digital_features = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
X = df[digital_features]

# Step 3: Scale digital data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Load autoencoder model with mse loss registered
autoencoder = load_model('models/sae_digital.h5', custom_objects={"mse": MeanSquaredError()})

# Step 5: Predict reconstruction (NO reshaping here!)
X_reconstructed = autoencoder.predict(X_scaled)

# Step 6: Calculate reconstruction error (MSE per sample)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

# Step 7: Train surrogate regressor to explain reconstruction error
regressor = RandomForestRegressor(n_estimators=50, random_state=42)
regressor.fit(X_scaled, reconstruction_error)

# Step 8: Initialize LIME explainer in regression mode
lime_explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=digital_features,
    mode='regression',
    random_state=42
)

# Step 9: Choose instance to explain
instance_index = 10  # Change to any index you want
instance = X_scaled[instance_index]

# Step 10: Explain reconstruction error prediction
lime_exp = lime_explainer.explain_instance(
    data_row=instance,
    predict_fn=regressor.predict,
    num_features=len(digital_features)
)

# Step 11: Print explanation
print(f"Explaining digital instance {instance_index}")
print(f"Predicted reconstruction error: {regressor.predict([instance])[0]:.6f}")
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight:.6f}")

# Step 12: Save LIME plot as PNG (if not in notebook)
fig = lime_exp.as_pyplot_figure()
fig.savefig(f"lime_digital_explanation_instance_{instance_index}.png", dpi=300, bbox_inches='tight')
print("LIME plot saved.")
