import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt
import tensorflow as tf
import keras
from keras.layers import Input,Dropout,Dense,LSTM,TimeDistributed,RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop(self.df.columns[0], axis=1)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.train_data = None
        self.test_data = None
        self.analog_train = None
        self.analog_test = None
        self.digital_train = None
        self.digital_test = None

    def preprocessing_df(self):
        # create a new column where 1 day before the timeframe as well as the timeframe indicated in the paper is labeled as unhealthy data i.e. 1
        # healthy data is set as 0
        self.df['is_anomaly'] = np.where(
            ((self.df['timestamp'] >= "2020-04-11 11:50:00") & (self.df['timestamp'] <= "2020-04-12 23:30:00")) |
            ((self.df['timestamp'] >= "2020-04-17 00:00:00") & (self.df['timestamp'] <= "2020-04-19 01:30:00")) |
            ((self.df['timestamp'] >= "2020-04-28 03:20:00") & (self.df['timestamp'] <= "2020-04-29 22:20:00")) |
            
            
            ((self.df['timestamp'] >= "2020-05-12 14:00:00") & (self.df['timestamp'] <= "2020-05-13 23:59:00")) |
            ((self.df['timestamp'] >= "2020-05-17 05:00:00") & (self.df['timestamp'] <= "2020-05-20 20:00:00")) |
            
            
            ((self.df['timestamp'] >= "2020-05-28 23:30:00") & (self.df['timestamp'] <= "2020-05-30 06:00:00")) |
            
            ((self.df['timestamp'] >= "2020-05-31 15:00:00") & (self.df['timestamp'] <= "2020-06-01 15:40:00")) |
            ((self.df['timestamp'] >= "2020-06-02 10:00:00") & (self.df['timestamp'] <= "2020-06-03 11:00:00")) |
            ((self.df['timestamp'] >= "2020-06-04 10:00:00") & (self.df['timestamp'] <= "2020-06-07 14:30:00")) |
            
            ((self.df['timestamp'] >= "2020-07-07 17:30:00") & (self.df['timestamp'] <= "2020-07-08 19:00:00")) |
            ((self.df['timestamp'] >= "2020-07-14 14:30:00") & (self.df['timestamp'] <= "2020-07-15 19:00:00")) |
            ((self.df['timestamp'] >= "2020-07-16 04:30:00") & (self.df['timestamp'] <= "2020-07-17 05:30:00"))
            ,
            1, 0
        )
        self.data_path = "data/dataset_train_processed.csv" # change this path accordingly when you want to change the file location
        self.df.to_csv(self.data_path)
        self.df = pd.read_csv(self.data_path)

    def preprocessing_autoencoder(self):
        # self.train_data = self.df[
        #     (self.df['timestamp'] >= "2022-01-01 06:00:00") & (self.df['timestamp'] <= "2022-02-28 02:00:00")]
        # self.test_data = self.df[
        #     (self.df['timestamp'] >= "2022-02-28 06:00:00")]
        self.train_data = self.df[
            (self.df['timestamp'] >= "2020-02-01 00:00:00") & (self.df['timestamp'] < "2020-03-20 11:59:59") & 
            (self.df['is_anomaly'] == 0)]
        self.test_data = self.df[
            (self.df['timestamp'] >= "2020-04-01 00:00:00") & (self.df['timestamp'] < "2020-07-31 11:59:59")]
        self.train_data.drop(self.train_data.columns[0], axis=1,
                             inplace=True)  # there is an additional unnecessary column created at index 0 from preprocessing.df that needs to be removed if preprocessing_df is called
        self.test_data.drop(self.test_data.columns[0], axis=1, inplace=True)
        scaler = StandardScaler() # standardising only the analog data, leaving the digital data as is
        self.analog_train = pd.DataFrame(scaler.fit_transform(self.train_data.iloc[:, 1:8]))
        self.digital_train = self.train_data.iloc[:, 8:16]
        print(self.test_data)
        print(max(self.df['timestamp']))
        self.analog_test = pd.DataFrame(scaler.transform(self.test_data.iloc[:, 1:8]))
        self.digital_test = self.test_data.iloc[:, 8:16]

def main():
    preprocessor = DataPreprocessor("data/MetroPT3(AirCompressor).csv") # change this to the file path of the train dataset
    preprocessor.preprocessing_df()

if __name__ == "__main__":
    main()