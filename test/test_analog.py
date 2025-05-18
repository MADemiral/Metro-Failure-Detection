
import numpy as np
import matplotlib.pyplot as plt
import pickle
from train.preprocess import DataPreprocessor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

np.set_printoptions(suppress=True, precision=10)

class AutoencoderResults:
    def __init__(self, train_data, test_data, analog_train, analog_test, digital_train, digital_test):
        self.train_data = train_data
        self.test_data = test_data
        self.analog_train = analog_train
        self.analog_test = analog_test
        self.digital_train = digital_train
        self.digital_test = digital_test
        self.train_pred = None
        self.test_pred = None
        self.threshold = None

        
    
    
    def compute_error_vae_analog(self):
        with open("vae_analog_train_pred", "rb") as file:
            self.train_pred = pickle.load(file)
        with open("vae_analog_test_pred", "rb") as file:
            self.test_pred = pickle.load(file)
        reconstruction_error = np.mean(np.square(self.train_pred - self.analog_train), axis=1) # can try mse/rmse/mae etc.
        q1 = np.percentile(reconstruction_error, 25)
        q3 = np.percentile(reconstruction_error, 75)
        self.threshold = q3 + 2 * (q3 - q1)
        # self.threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)  # can be adjusted to find the optimal threshold
        test_error = np.mean(np.square(self.test_pred - self.analog_test), axis=1)
        filtered_error = self.low_pass_filter(test_error,
                                         alpha=0.1)  # can be adjusted to find optimal alpha
        predictions = [1 if error > self.threshold else 0 for error in filtered_error]
        self.test_data["predicted_output"] = predictions
    
    
    def low_pass_filter(self, error, alpha): # test_error is passed through low_pass_filter to improve evaluation metrics
        error = error.reset_index(drop = True)
        output = [error[0]]  # initialize the output with the first input value
        for i in range(1, len(error)):
            output.append(output[i - 1] + alpha * (error[i] - output[i - 1]))
        return output

    def create_sequence(self, anomalies):  # check if anomaly data points appear consecutively
    # if a point appears by itself, it has a high chance of simply being random noise rather than a part of any failure
    # returns index of anomalies
        tmp1 = []
        tmp2 = []
        lst = []
        for i in range(len(anomalies)):
            if anomalies[i] == 1:
                tmp1.append(i)
        for j in range(len(tmp1) - 1):
            if tmp1[j] + 1 == tmp1[j + 1]:  # check if the next index is the same as the previous index + 1 (eg. 501 == 500 + 1)
                tmp2.append(tmp1[j])
            if tmp1[j + 1] != tmp1[j] + 1:
                if len(tmp2) >= 2:  # sequences have to be x seconds or more - can be adjusted to affect precision/recall
                    tmp2.append(tmp1[j])
                    lst.extend(tmp2)
                tmp2 = []
        return lst
        
    
        
        
    def plot_cm_VAE_Analog(self):
        cm = confusion_matrix(self.test_data["is_anomaly"], self.test_data["predicted_output"])
        tp = cm[1, 1]
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        print(classification_report(self.test_data["is_anomaly"], self.test_data["predicted_output"]))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
        disp.plot(values_format='d', cmap="Reds")
        plt.title("Variational Autoencoder - Analog (alpha=0.05)\nPrecision={:.2f}, Recall={:.2f}, F1 Score={:.2f}".format(precision, recall, f1_score))
        plt.savefig("sae_vae_plots/vae_analog_confusion_matrix.png")  # save the plot as a .png file
        plt.show()


def main():
    preprocessor = DataPreprocessor("data/dataset_train_processed.csv")
    preprocessor.preprocessing_autoencoder()
    autoencoder_results = AutoencoderResults(preprocessor.train_data, preprocessor.test_data, preprocessor.analog_train,
                                             preprocessor.analog_test, preprocessor.digital_train, preprocessor.digital_test)
    autoencoder_results.compute_error_vae_analog()
    autoencoder_results.plot_cm_VAE_Analog()


if __name__ == "__main__":
    main()