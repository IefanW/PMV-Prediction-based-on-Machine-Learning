import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from keras.callbacks import ReduceLROnPlateau,EarlyStopping

# read data
dataset_train = pd.read_csv("LSTM_data/data_train.csv")
training_set = dataset_train.iloc[:,5:].values

# data preprocessing
mms = MinMaxScaler(feature_range=(0,1))
training_set_scaled = mms.fit_transform(training_set)

# create training data through timesteps
x_train = []
y_train = []
for i in range(24, 8760):
    x_train.append(training_set_scaled[i-24:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# callbacks definition
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
callback_list = [reduce_lr]

start = time.time()
# build LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 32, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = regressor.fit(x_train, y_train, epochs = 100, batch_size = 24,callbacks=callback_list)

# testing prediction
dataset_test = pd.read_csv('/Users/iefan_wey/Desktop/毕业设计/BPNN/Implementation/LSTM_testing_data/data1_1.csv')
y_test = dataset_test.iloc[24:, 5:].values

dataset_total = pd.concat((dataset_train['PMV'], dataset_test['PMV']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(y_test) - 24:].values

inputs = inputs.reshape(-1,1)
inputs = mms.transform(inputs)
X_test = []
for i in range(24, 8760):
    X_test.append(inputs[i-24:i, 0])
X_test1 = np.array(X_test)

print(X_test1[0])
print(X_test1[1])
X_test = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))

y_pred = regressor.predict(X_test)
predicted_pmv = mms.inverse_transform(y_pred)

end = time.time()
# evaluation
loss = mean_squared_error(y_test,predicted_pmv)
mae = mean_absolute_error(y_test,predicted_pmv)
score = r2_score(y_test,predicted_pmv)


print("mse is " + str(loss))
print("rmse is " + str(loss**0.5))
print("mae is " + str(mae))
print("r2 is "+ str(score))
print("fitting time is " +str(end-start))


# visualization
epochs=range(len(history.history['loss']))
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.title('Accuracy of the model')

# plt.plot(y_test, color = 'blue', label = 'Real PMV')
# plt.plot(predicted_pmv, color = 'red', label = 'Predicted PMV')
plt.xlabel('Time')
plt.ylabel('PMV')
plt.legend()
plt.show()