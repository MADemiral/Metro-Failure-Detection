
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

file_path = "data/MetroPT3(AirCompressor).csv"

df = pd.read_csv(file_path)

# Show the first few rows
df.head()

"""Check NULL"""

df.isna().sum()

"""Dataset Info"""

print("\nDataset Info:")
df.info()

"""Summary Statistics"""

print("\nSummary Statistics:")
df.describe()

"""Drop unnecessary columns and convert timestampt to datetime"""

df = df.drop(columns=['Unnamed: 0'])

df['timestamp'] = pd.to_datetime(df['timestamp'])

"""Scale Sensor Data"""

sensordata_cols = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current', 'COMP',
    'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses'
]
# scale sensordata columns
scaler = StandardScaler()
df[sensordata_cols] = scaler.fit_transform(df[sensordata_cols])

"""segment data"""

segment_length = 60

df.set_index('timestamp', inplace=True)

segmented_data = df.resample(f'{segment_length}S').agg({
    'TP2': ['mean', 'std', 'min', 'max'],
    'TP3': ['mean', 'std', 'min', 'max'],
    'H1': ['mean', 'std', 'min', 'max'],
    'DV_pressure': ['mean', 'std', 'min', 'max'],
    'Reservoirs': ['mean', 'std', 'min', 'max'],
    'Oil_temperature': ['mean', 'std', 'min', 'max'],
    'Motor_current': ['mean', 'std', 'min', 'max'],
    'COMP': ['mean', 'std'],
    'DV_eletric': ['mean', 'std'],
    'Towers': ['mean', 'std'],
    'MPG': ['mean', 'std'],
    'LPS': ['mean', 'std'],
    'Pressure_switch': ['mean', 'std'],
    'Oil_level': ['mean', 'std'],
    'Caudal_impulses': ['mean', 'std']
})
#flatten
segmented_data.columns = ['_'.join(col).strip() for col in segmented_data.columns.values]

# Drop any rows with missing values due to resampling
segmented_data.dropna(inplace=True)

# Reset index for further processing
segmented_data.reset_index(inplace=True)

# Print the shape and a preview of the segmented data
print(f"Segmented data shape: {segmented_data.shape}")
segmented_data.head()

"""Order by timestampt split train test"""

# Prepare data for training
features = segmented_data.drop(columns=['timestamp']).values
train_data, val_data = train_test_split(features, test_size=0.2, random_state=42)

"""# Train

k Fold
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import pickle  # For saving sklearn models

input_dim = features.shape[1]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

"""train sparse auto encoder"""

for fold, (train_index, val_index) in enumerate(kf.split(features)):
    print(f"\n🔁 Fold {fold+1} - Training Sparse Autoencoder")

    X_train, X_val = features[train_index], features[val_index]

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(input_layer)
    bottleneck = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(bottleneck)
    output_layer = Dense(input_dim, activation='linear')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=256,
        validation_data=(X_val, X_val),
        shuffle=True,
        verbose=0
    )

    model_path = f"autoencoder_fold{fold+1}.h5"
    autoencoder.save(model_path)
    print(f"✅ Saved: {model_path}")

"""Train Isolation Forest"""

for fold, (train_index, _) in enumerate(kf.split(features)):
    print(f"\n🔁 Fold {fold+1} - Training Isolation Forest")

    X_train = features[train_index]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)

    model_path = f"isolation_forest_fold{fold+1}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {model_path}")

"""Train One-Class SVM"""

for fold, (train_index, _) in enumerate(kf.split(features)):
    print(f"\n🔁 Fold {fold+1} - Training One-Class SVM")

    X_train = features[train_index]
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
    model.fit(X_train)

    model_path = f"ocsvm_fold{fold+1}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {model_path}")

"""Train Local Outlier Factor"""

for fold, (train_index, _) in enumerate(kf.split(features)):
    print(f"\n🔁 Fold {fold+1} - Training Local Outlier Factor")

    X_train = features[train_index]
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
    model.fit(X_train)

    model_path = f"lof_fold{fold+1}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {model_path}")

"""Elliptic Envelope"""

for fold, (train_index, _) in enumerate(kf.split(features)):
    print(f"\n🔁 Fold {fold+1} - Training Elliptic Envelope")

    X_train = features[train_index]
    model = EllipticEnvelope(contamination=0.05, random_state=42)
    model.fit(X_train)

    model_path = f"elliptic_fold{fold+1}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved: {model_path}")

"""Evaluate and Save Models"""
