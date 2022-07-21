#!/usr/bin/env python
# coding: utf-8

# Data wrangling
import pandas as pd 

# Visualization
import matplotlib.pyplot as plt 
import seaborn as sns 

# Date wrangling
import datetime

# Math operations
import numpy as np

# Random sampling
import random

# Keras API 
from tensorflow import keras

# Deep learning 
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Concatenate, SimpleRNN, Masking, Flatten
from keras import losses
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal

#json to save datas
import json


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of 
    deep learning models 
    """
    # Extracting the number of features that are passed from the array 
    n_features = ts.shape[1]
    
    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an RNN input shape 
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y

class Neural_Network_Model():
    
    def __init__(
        self, 
        X, 
        Y, 
        n_outputs,
        n_lag,
        n_ft,
        n_layer,
        batch,
        epochs, 
        lr,
        Xval=None,
        Yval=None,
        mask_value=-999.0,
        min_delta=0.001,
        patience=5,
        verbose=2
    ):
        lstm_input = Input(shape=(n_lag, n_ft))
      # Series signal 
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)
        x = Dense(n_outputs)(lstm_layer)
        
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch 
        self.epochs = epochs
        self.n_layer=n_layer
        self.lr = lr 
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.mask_value = mask_value
        self.min_delta = min_delta
        self.patience = patience
        self.verbose=verbose
        

    def trainCallback(self):
        return EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta)

    def train(self):
        # Getting the untrained model 
        empty_model = self.model
        
        # Initiating the optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model
        empty_model.compile(loss=losses.MeanAbsoluteError(), optimizer=optimizer)

        
        history = empty_model.fit(
            self.X, 
            self.Y, 
            epochs=self.epochs, 
            batch_size=self.batch,
            shuffle=False,
            validation_data=(self.Xval, self.Yval),
            verbose=self.verbose,
            callbacks=[self.trainCallback()]
            )
            
        
        # Saving to original model attribute in the class
        self.model = empty_model
        
        # Returning the training history
        return history
    
    def predict(self, X):
        return self.model.predict(X)

##################################################import data##################################################

#2019 file
df1=pd.read_csv('eplusout_2019.csv',sep=',',decimal=',',index_col=0, low_memory=False)

df1 = df1.filter(['DistrictCooling:Facility [J](Hourly)','Electricity:Facility [J](Hourly)','DistrictHeating:Facility [J](Hourly)'])
df1=df1.reset_index()
data1 = df1['Date/Time'].tolist()

for i in range( 0, len(data1)):
    data1[i] = data1[i].replace( str(data1[i]), '19/'+ str(data1[i]).strip())

new_df1={}
new_df1['Date/Time']=data1
new_df1['DistrictCooling:Facility [J](Hourly)']= df1['DistrictCooling:Facility [J](Hourly)'].tolist()
new_df1=pd.DataFrame(new_df1)
new_df1.set_index("Date/Time", inplace = True)

#2020 file

df2=pd.read_csv('eplusout_2020.csv',sep=',',decimal=',',index_col=0, low_memory=False)
df2 = df2.filter(['DistrictCooling:Facility [J](Hourly)','Electricity:Facility [J](Hourly)','DistrictHeating:Facility [J](Hourly)'])
df2=df2.reset_index()
data2 = df2['Date/Time'].tolist()

for i in range( 0, len(data2)):
    data2[i] = data2[i].replace( str(data2[i]), '20/'+ str(data2[i]).strip())

new_df2={}
new_df2['Date/Time']=data2
new_df2['DistrictCooling:Facility [J](Hourly)']= df2['DistrictCooling:Facility [J](Hourly)'].tolist()
new_df2=pd.DataFrame(new_df2)
new_df2.set_index("Date/Time", inplace = True)

#2021 file

df3=pd.read_csv('eplusout_2021.csv',sep=',',decimal=',',index_col=0, low_memory=False)
df3 = df3.filter(['DistrictCooling:Facility [J](Hourly)','Electricity:Facility [J](Hourly)','DistrictHeating:Facility [J](Hourly)'])
df3=df3.reset_index()
data3 = df3['Date/Time'].tolist()

for i in range( 0, len(data3)):
    data3[i] = data3[i].replace( str(data3[i]), '21/'+ str(data3[i]).strip())

new_df3={}
new_df3['Date/Time']=data3
new_df3['DistrictCooling:Facility [J](Hourly)']= df3['DistrictCooling:Facility [J](Hourly)'].tolist()
new_df3=pd.DataFrame(new_df3)
new_df3.set_index("Date/Time", inplace = True)


dftot=pd.concat([new_df1, new_df2, new_df3 ])
d = dftot.filter(['DistrictCooling:Facility [J](Hourly)'])
d=d.reset_index()


# Types of columns
d['DistrictCooling:Facility [J](Hourly)']= d['DistrictCooling:Facility [J](Hourly)'].astype(float)
d.dtypes

# Features used in models
features = ['DistrictCooling:Facility [J](Hourly)']

# Aggregating to hourly level
d = d.groupby('Date/Time', as_index=False)[features].mean()
d[features].describe()

####################################### Hyper parameters #######################################

# Number of lags (hours back) to use for models
lag = 72

# Steps ahead to forecast 
n_ahead = 3

# Share of obs in testing 
test_share = 0.34

# Epochs for training
epochs = 50

# Batch size 
batch_size = 32

# Learning rate
lr = 0.001

# Number of neurons in LSTM layer
n_layer = 50

# The features used in the modeling 
features_final = ['DistrictCooling:Facility [J](Hourly)']


########################################## Preparing data ###########################################
# Subseting only the needed columns 
ts = d[features_final]

nrows = ts.shape[0]

# Spliting into train and test sets
train = ts[0:int(nrows * (1 - test_share))].astype('float')
test = ts[int(nrows * (1 - test_share)):].astype('float')

# Scaling the data 
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

# Creating the final scaled frame 
ts_s = pd.concat([train, test])

X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
n_ft = X.shape[2]

# Spliting into train and test sets 
Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]

############################# Define the model ############################
model = Neural_Network_Model(
    X=Xtrain,
    Y=Ytrain,
    n_outputs=n_ahead,
    n_lag=lag,
    n_ft=n_ft,
    n_layer=n_layer,
    batch=batch_size,
    epochs=epochs, 
    lr=lr,
    Xval=Xval,
    Yval=Yval,
)

model.model.summary()
history = model.train()
loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

######################## loss function plot ########################
n_epochs = range(len(loss))
plt.figure(figsize=(9, 7))
plt.plot(n_epochs, loss,label='Training loss', color='blue')
plt.plot(n_epochs, val_loss,label='Validation loss', color='red')
plt.legend(loc=0)
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.show()

# Comparing the forecasts with the actual values
yhat = [x[0] for x in model.predict(Xval)]
y = [y[0] for y in Yval]

# Creating the frame to store both predictions
days = d['Date/Time'].values[-len(y):]

frame = pd.concat([
    pd.DataFrame({'day': days, 'temp': y, 'type': 'original'}),
    pd.DataFrame({'day': days, 'temp': yhat, 'type': 'forecast'})
])

# Creating the unscaled values column
frame['temp_absolute'] = [(x* train_std['DistrictCooling:Facility [J](Hourly)'] )+ train_mean['DistrictCooling:Facility [J](Hourly)']  for x in frame['temp']] # ---> cooling

# Computation of temp results
temp_res = frame.pivot_table(index='day', columns='type')
temp_res.columns = ['_'.join(x).strip() for x in temp_res.columns.values]
temp_res['res'] = temp_res['temp_absolute_original'] - temp_res['temp_absolute_forecast']
temp_res['res_abs'] = [abs(x) for x in temp_res['res']]
temp_res.res_abs.describe()

# Forecasting on all the samples in the validation set 
forecast = model.predict(Xval)

# Calculating the total average absolute error 
error = 0 
n=0
residuals = []
error_mse=0

for i in range(Yval.shape[0]):
    true = Yval[i]
    hat = forecast[i]
    n += len(true)
    
    true = np.asarray([(x * train_std['DistrictCooling:Facility [J](Hourly)']) + train_mean['DistrictCooling:Facility [J](Hourly)'] for x in true])
    hat = np.asarray([(x * train_std['DistrictCooling:Facility [J](Hourly)']) + train_mean['DistrictCooling:Facility [J](Hourly)'] for x in hat])
       
    residual = true - hat
    residuals.append(residual)
    
    error += np.sum([abs(x) for x in true - hat]) 
    error_mse += np.sum([pow(x,2) for x in true - hat]) 

# Flattening the list of arrays of residuals
residuals = np.asarray(residuals).flatten().tolist()
abs_residuals = [abs(x) for x in residuals]
mse_residuals = [pow(x,2) for x in residuals]

######################################## FINAL RESULTS ##########################################################

Median_absoulte_error=round(np.median(abs_residuals), 2)
Mean_absoulte_error= round(np.mean(abs_residuals), 2)
Mean_square_error=round(np.mean(mse_residuals), 2)
Root_mean_square_error=round(np.sqrt(np.mean(mse_residuals)), 2)

print("**************************FINAL RESULTS*************************************")
print("Median absoulte error: " + str( Median_absoulte_error )+"°C")
print("Mean absoulte error: "+ str(Mean_absoulte_error)+"°C")
print("Mean square error: " + str(Mean_square_error) + "°C")
print("Root mean squareerror: " + str(Root_mean_square_error) + "°C")

################################### cooling annual consumption plot #######################################
plt.figure(figsize=(12, 12))
plt.plot(temp_res.index, temp_res.temp_absolute_original, color='blue', label='original')
plt.plot(temp_res.index, temp_res.temp_absolute_forecast, color='red', label='forecast', alpha=0.6)
plt.title('Cooling - Annual energy consumption')
plt.ylabel('Energy Consumption [J]')
plt.xticks(np.arange(0,8760,876), rotation=45)
plt.legend()
plt.show()

###################################### saving data in a json ############################################
dict1 = { "setting": { 
                        
                        "Number of lags (hours back) to use for models": lag,                        
                        "Steps ahead to forecast" : n_ahead,                         
                        "Share of obs in testing" : test_share,
                        "Epochs for training" : epochs,
                        " Batch size" : batch_size,                         
                        "Learning rate" : lr,                         
                        "Number of neurons in LSTM layer" :  n_layer 
                            },
                           
           "test": { 
                            "RMSE": Root_mean_square_error, 
                            "MSE": Mean_square_error,  
                            "MAE": Mean_absoulte_error
                        }
                       
                        } 

# the json file where the output must be stored 
out_file = open("settings_and_results_power_consumption_onlycooling.json", "a") 

json.dump(dict1, out_file, indent = 6) 

out_file.close()
