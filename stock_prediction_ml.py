# -*- coding: utf-8 -*-

#Stock Price prediction System using Machine Learning
#By- Saksham Prajapati

# Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import precision_score, accuracy_score

#Loading Dataset

#url='AXISBANK.csv'
df=pd.read_csv('AXISBANK.csv')

df["Date"]=pd.to_datetime(df.Date,format="%d-%m-%Y")
df.index=df['Date']


scaler=MinMaxScaler(feature_range=(0,1))

data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]


new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

#---> Spliting Data for Training & Testing
final_dataset=new_dataset.values

train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

#---> Model Training
"""
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
"""

# Initialising the RNN
lstm_model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train_data.shape[1],1)))
lstm_model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))

#Validation Data for model
inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

# Adding the output layer
lstm_model.add(Dense(units = 1))

# Compiling the RNN
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
lstm_model.fit(x_train_data,y_train_data, epochs = 100, batch_size = 50)

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)


X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price=lstm_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

#---> Saving Trained Model
lstm_model.save("saved_lstm_model.h5")

train_data=new_dataset[:987]
valid_data=new_dataset[987:]
valid_data['Predictions']=predicted_closing_price


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(valid_data['Close'], valid_data['Predictions'])

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(valid_data['Close'], valid_data['Predictions'])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")