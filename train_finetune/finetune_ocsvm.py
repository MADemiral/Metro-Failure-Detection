import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from train.preprocess import DataPreprocessor
import os

def train_and_finetune_ocsvm(preprocessor: DataPreprocessor, model_path="models/ocsvm_analog.pkl"):
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../ocsvm", exist_ok=True)

    # Get preprocessed data
    analog_train = preprocessor.analog_train
    analog_test = preprocessor.analog_test
    test_labels = preprocessor.test_data['is_anomaly'].values.astype(int)

    # Train OCSVM from scratch
    model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    model.fit(analog_train)

    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {model_path}")

    # Predict
    preds = model.predict(analog_test)
    preds_binary = [0 if p == 1 else 1 for p in preds]  # 1 = anomaly

    # Evaluation
    cm = confusion_matrix(test_labels, preds_binary)
    cr = classification_report(test_labels, preds_binary, output_dict=True)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(test_labels, preds_binary))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("ocsvm/ocsvm_confusion_matrix.png")
    plt.close()

    # Plot classification report as bar plot
    plt.figure(figsize=(8, 5))
    report_df = {
        "Precision": [cr['0']['precision'], cr['1']['precision']],
        "Recall": [cr['0']['recall'], cr['1']['recall']],
        "F1-Score": [cr['0']['f1-score'], cr['1']['f1-score']],
    }
    labels = ['Normal (0)', 'Anomaly (1)']
    x = np.arange(len(labels))
    width = 0.25

    for idx, (metric, values) in enumerate(report_df.items()):
        plt.bar(x + idx * width, values, width=width, label=metric)

    plt.xticks(x + width, labels)
    plt.ylim(0, 1.1)
    plt.title("Classification Report")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ocsvm/ocsvm_classification_report.png")
    plt.close()

    print("Plots saved to 'plots/' directory.")
    return preds_binary

def main():
    preprocessor = DataPreprocessor("data/dataset_train_processed.csv")
    preprocessor.preprocessing_autoencoder()
    train_and_finetune_ocsvm(preprocessor)

if __name__ == "__main__":
    main()
