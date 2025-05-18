

import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf# -*- coding: utf-8 -*-

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

"""label known anomaly"""

df['is_anomaly'] = np.where(
            ((df['timestamp'] >= "2020-04-11 11:50:00") & (df['timestamp'] <= "2020-04-12 23:30:00")) |
            ((df['timestamp'] >= "2020-04-17 00:00:00") & (df['timestamp'] <= "2020-04-19 01:30:00")) |
            ((df['timestamp'] >= "2020-04-28 03:20:00") & (df['timestamp'] <= "2020-04-29 22:20:00")) |
            
            
            ((df['timestamp'] >= "2020-05-12 14:00:00") & (df['timestamp'] <= "2020-05-13 23:59:00")) |
            ((df['timestamp'] >= "2020-05-17 05:00:00") & (df['timestamp'] <= "2020-05-20 20:00:00")) |
            
            
            ((df['timestamp'] >= "2020-05-28 23:30:00") & (df['timestamp'] <= "2020-05-30 06:00:00")) |
            
            ((df['timestamp'] >= "2020-05-31 15:00:00") & (df['timestamp'] <= "2020-06-01 15:40:00")) |
            ((df['timestamp'] >= "2020-06-02 10:00:00") & (df['timestamp'] <= "2020-06-03 11:00:00")) |
            ((df['timestamp'] >= "2020-06-04 10:00:00") & (df['timestamp'] <= "2020-06-07 14:30:00")) |
            
            ((df['timestamp'] >= "2020-07-07 17:30:00") & (df['timestamp'] <= "2020-07-08 19:00:00")) |
            ((df['timestamp'] >= "2020-07-14 14:30:00") & (df['timestamp'] <= "2020-07-15 19:00:00")) |
            ((df['timestamp'] >= "2020-07-16 04:30:00") & (df['timestamp'] <= "2020-07-17 05:30:00"))
            ,
            1, 0
        )
data_path = "dataset_train_processed.csv" # change this path accordingly when you want to change the file location
df.to_csv(data_path)
df = pd.read_csv(data_path)



train_data = df[
(df['timestamp'] >= "2020-02-01 00:00:00") & (df['timestamp'] < "2020-04-11 11:49:59")]
val_data = df[
(df['timestamp'] >= "2020-04-11 11:50:00") & (df['timestamp'] < "2020-05-30 11:59:59")]
train_data.drop(train_data.columns[0], axis=1,
                inplace=True)  # there is an additional unnecessary column created at index 0 from preprocessing.df that needs to be removed if preprocessing_df is called
val_data.drop(val_data.columns[0], axis=1, inplace=True)
scaler = StandardScaler() # standardising only the analog data, leaving the digital data as is
analog_train = pd.DataFrame(scaler.fit_transform(train_data.iloc[:, 1:8]))
digital_train = train_data.iloc[:, 8:16]
analog_test = pd.DataFrame(scaler.transform(val_data.iloc[:, 1:8]))
digital_test = val_data.iloc[:, 8:16]


"""# Train

"""

import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


os.makedirs("models", exist_ok=True)  # Ensure directory exists

# ---------------------------
# SAE Autoencoder (for analog and digital)
# ---------------------------
def build_sae_model(input_dim, output_activation='linear'):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    bottleneck = Dense(6, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(bottleneck)
    decoded = Dense(32, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation=output_activation)(decoded)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_sae(X, name):
    model = build_sae_model(X.shape[1], output_activation='linear')
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X, X, epochs=50, batch_size=64, validation_split=0.2, callbacks=[es])
    save_model(model, f"models/sae_{name}.h5")

# ---------------------------
# OC-SVM
# ---------------------------
def train_ocsvm(X, name):
    model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    model.fit(X)
    joblib.dump(model, f"models/ocsvm_{name}.pkl")

# ---------------------------
# Isolation Forest
# ---------------------------
def train_isolation_forest(X, name):
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X)
    joblib.dump(model, f"models/isoforest_{name}.pkl")

# ---------------------------
# Elliptic Envelope
# ---------------------------
def train_elliptic_envelope(X, name):
    model = EllipticEnvelope(contamination=0.01)
    model.fit(X)
    joblib.dump(model, f"models/elliptic_{name}.pkl")

# ---------------------------
# Local Outlier Factor (Only for prediction, not "fit then predict")
# ---------------------------
def train_lof(X, name):
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.01)
    model.fit(X)
    joblib.dump(model, f"models/lof_{name}.pkl")

# ---------------------------
# Train All Models
# ---------------------------
def train_all_models(analog_data, digital_data):
    train_sae(analog_data, "analog")
    train_sae(digital_data, "digital")
    """train_ocsvm(analog_data, "analog")
    train_isolation_forest(analog_data, "analog")
    train_elliptic_envelope(analog_data, "analog")
    train_lof(analog_data, "analog")

    print("Training for Digital Sensors...")

    train_ocsvm(digital_data, "digital")
    train_isolation_forest(digital_data, "digital")
    train_elliptic_envelope(digital_data, "digital")
    train_lof(digital_data, "digital")"""

# Example usage (after preparing analog_train and digital_train):
analog_train_np = analog_train
digital_train_np = digital_train

train_all_models(analog_train_np, digital_train_np)

import os
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# === Create output directory ===
os.makedirs("evaluation_plots", exist_ok=True)

# === Confusion matrix plot function ===
def plot_confusion_matrix(y_true, y_pred, model_name, sensor_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Anomaly"], 
                yticklabels=["Normal", "Anomaly"])
    plt.title(f'Confusion Matrix - {model_name} ({sensor_type})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"evaluation_plots/confusion_matrix_{model_name}_{sensor_type}.png")
    plt.close()

# === Evaluation report function ===
def evaluate_model(y_true, y_pred, model_name, sensor_type):
    print(f"\nEvaluation - {model_name} ({sensor_type})")
    print(classification_report(y_true, y_pred, digits=4))
    plot_confusion_matrix(y_true, y_pred, model_name, sensor_type)

# === Autoencoder prediction ===
def get_predictions_autoencoder(model_path, X, threshold=None):
    model = load_model(model_path, custom_objects={"mse": MeanSquaredError()})
    recon = model.predict(X)
    mse = np.mean(np.square(X - recon), axis=1)
    if threshold is None:
        threshold = np.percentile(mse, 99)  # top 1% as anomalies
    return (mse > threshold).astype(int)

# === Scikit-learn based prediction ===
def get_predictions_sklearn(model_path, X):
    model = joblib.load(model_path)
    preds = model.predict(X)
    return np.where(preds == -1, 1, 0)  # Convert -1 to 1 (anomaly), 1 to 0 (normal)

# === Combined evaluation for all models and sensor types ===
def evaluate_all(val_analog, val_digital, true_labels):
    models = {
        "sae": get_predictions_autoencoder,
        "ocsvm": get_predictions_sklearn,
        "isoforest": get_predictions_sklearn,
        "elliptic": get_predictions_sklearn,
        "lof": get_predictions_sklearn
    }

    for sensor_type, val_data in [("analog", val_analog), ("digital", val_digital)]:
        for model_name, predictor in models.items():
            model_file = f"models/{model_name}_{sensor_type}.h5" if model_name == "sae" else f"models/{model_name}_{sensor_type}.pkl"
            print(f"Evaluating {model_name.upper()} model for {sensor_type} sensor...")
            preds = predictor(model_file, val_data)
            evaluate_model(true_labels, preds, model_name, sensor_type)



# Prepare numpy arrays
analog_val_np = analog_test.values
digital_val_np = digital_test.values
true_labels = val_data['is_anomaly'].values

# === Run the evaluation ===
evaluate_all(analog_val_np, digital_val_np, true_labels)
