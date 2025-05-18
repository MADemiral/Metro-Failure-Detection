import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from datetime import timedelta

# Step 1: Load preprocessed data
df = pd.read_csv('data/dataset_train_processed.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

start_original = pd.Timestamp("2020-04-11 11:50:00")
end_original = pd.Timestamp("2020-04-12 23:30:00")

# Expand 6 hours before and after
start_window = start_original - timedelta(hours=6)
end_window = end_original + timedelta(hours=6)

df = df[(df['timestamp'] >= start_window) & (df['timestamp'] <= end_window)]

# Step 2: Define analog features
analog_features = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Motor_current', 'Oil_temperature']

# Step 3: Select analog data and scale
X = df[analog_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Load One-Class SVM model and create surrogate labels
with open('models/ocsvm_analog.pkl', 'rb') as f:
    ocsvm_model = pickle.load(f)

pseudo_labels = (ocsvm_model.predict(X_scaled) == -1).astype(int)

# Step 5: Train surrogate RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_scaled, pseudo_labels)

# Step 6: Initialize LIME explainer
lime_explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=analog_features,
    class_names=['Normal', 'Anomaly'],
    mode='classification',
    random_state=42
)

# Step 7: Choose instance index to explain
instance_index = 10  # Change this index as needed
instance = X_scaled[instance_index]

# Step 8: Explain prediction
lime_exp = lime_explainer.explain_instance(
    data_row=instance,
    predict_fn=clf.predict_proba,
    num_features=len(analog_features)
)

# Step 9: Print explanation text
print(f"Explaining instance {instance_index}:")
print(f"Prediction probabilities: {clf.predict_proba([instance])[0]}")
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight:.4f}")

# Optional: Visualize explanation in notebook (if using Jupyter)
# lime_exp.show_in_notebook()

# Optional: Save plot to PNG (if not in notebook)
fig = lime_exp.as_pyplot_figure()
fig.savefig(f"lime_explanation_instance_{instance_index}.png", dpi=300, bbox_inches='tight')
print("LIME explanation plot saved.")
