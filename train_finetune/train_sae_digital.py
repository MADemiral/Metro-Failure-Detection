import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import keras.backend as K
from train.preprocess import DataPreprocessor
from tensorflow import keras
from keras.models import Model, save_model
from keras.layers import Conv1D, Conv1DTranspose, Lambda, Reshape, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from tensorflow.python.framework.ops import disable_eager_execution


class AutoencoderModels:

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = None
        self.hist = None

    def leaky_relu(self, x):  # to be used as the activation function of the output layer for the analog models; greatly reduces the loss value
        return tf.nn.leaky_relu(x, alpha=0.01)

    def sae_analog(self):  # both analog and digital sensors have 8 features/columns
        # Encoder portion
        input_layer = keras.Input(shape=(1, self.train_data.shape[1])) # shape of input is (no. of timesteps = 1, no. of features = 8) - same for all models
        hidden_1 = Conv1D(filters=32, padding="same", kernel_size=3, activation='relu')(input_layer)
        hidden_2 = Conv1D(filters=16, padding="same", kernel_size=3, activation='relu')(hidden_1)
        bottleneck = Conv1D(filters=6, padding="same", kernel_size=3, activation="relu", activity_regularizer=keras.regularizers.l1_l2(l1=0.005, l2=0.1))(hidden_2)
        hidden_3 = Conv1DTranspose(filters=16, padding="same", kernel_size=3, activation='relu')(bottleneck)
        hidden_4 = Conv1DTranspose(filters=32, padding="same", kernel_size=3, activation='relu')(hidden_3)
        output_layer = Conv1DTranspose(filters=self.train_data.shape[1], padding="same", kernel_size=3,
                              activation=self.leaky_relu)(
            hidden_4)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse') # loss function can also be set to RMSE, MAE etc. for parameter tuning
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        hist = model.fit(np.expand_dims(self.train_data, 1), np.expand_dims(self.train_data, 1), validation_split=0.2,
                         epochs=50, callbacks=[callback], batch_size=60)
        save_model(model, "sae_analog.h5")
        self.model = model
        self.hist = hist

    def sae_digital(self):
        input_layer = keras.Input(shape=(1, self.train_data.shape[1]))
        hidden_1 = Conv1D(filters=32, padding="same", kernel_size=3, activation='relu')(input_layer)
        hidden_2 = Conv1D(filters=16, padding="same", kernel_size=3, activation='relu')(hidden_1)
        bottleneck = Conv1D(filters=6, padding="same", kernel_size=3, activation="relu")(hidden_2)

        # Decoder Portion
        hidden_3 = Conv1DTranspose(filters=16, padding="same", kernel_size=3, activation='relu')(bottleneck)
        hidden_4 = Conv1DTranspose(filters=32, padding="same", kernel_size=3, activation='relu')(hidden_3)
        output_layer = Conv1DTranspose(filters=self.train_data.shape[1], padding="same", kernel_size=3, activation="sigmoid")(
            hidden_4)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        hist = model.fit(np.expand_dims(self.train_data, 1), np.expand_dims(self.train_data, 1), validation_split=0.2,
                         epochs=12, batch_size=64)
        save_model(model, "sae_digital.h5")
        self.model = model
        self.hist = hist



    def autoencoder_predict_digital(self):
        train_pred = self.model.predict(np.expand_dims(self.train_data, 1))
        print("Shape of train_pred before reshaping:", train_pred.shape)
        
        train_pred = np.reshape(train_pred, (len(train_pred), 8))
    
        test_pred = self.model.predict(np.expand_dims(self.test_data, 1))
        print("Shape of test_pred before reshaping:", test_pred.shape)
        test_pred = np.reshape(test_pred, (len(test_pred), 8))
        return train_pred, test_pred
    
    def autoencoder_predict_analog(self):
        train_pred = self.model.predict(np.expand_dims(self.train_data, 1))
        print("Shape of train_pred before reshaping:", train_pred.shape)
        
        train_pred = np.reshape(train_pred, (len(train_pred), 7))
    
        test_pred = self.model.predict(np.expand_dims(self.test_data, 1))
        print("Shape of test_pred before reshaping:", test_pred.shape)
        test_pred = np.reshape(test_pred, (len(test_pred), 7))
        return train_pred, test_pred

    def plot_losses(self):
        plt.plot(self.hist.history["loss"])
        plt.plot(self.hist.history["val_loss"])
        plt.title("Training/Validation Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["training_loss", "validation_loss"])
        plt.savefig("sae_vae_plots/sae_digital_loss_plot.png") # save the plot as a .png file
        plt.show()


def main():
    preprocessor = DataPreprocessor("data/dataset_train_processed.csv") # change according to where your data is stored

    preprocessor.preprocessing_autoencoder()
    autoencoder = AutoencoderModels(preprocessor.digital_train, preprocessor.digital_test) # comment out depending on whether digital/analog data is used
    autoencoder.sae_digital()
    train_pred, test_pred = autoencoder.autoencoder_predict_digital()
    with open("models/sae_digital_train_pred", "wb") as file: # name of file path can change depending on where you want to save it
        pickle.dump(train_pred, file)
    with open("models/sae_digital_test_pred", "wb") as file: # similarly, this file path can be changed
        pickle.dump(test_pred, file)
    print(autoencoder.model.summary())
    autoencoder.plot_losses()

if __name__ == "__main__":
    main()