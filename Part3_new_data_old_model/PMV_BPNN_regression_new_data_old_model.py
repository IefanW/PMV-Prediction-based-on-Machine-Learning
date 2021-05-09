import pandas as pd
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation, Dropout

input_file = "../Concatenated_data/filtered_data.csv"
data = pd.read_csv(input_file,encoding='gbk')
data = shuffle(data)
#data = data[['Outsides_tem','Temperature','Related_humidity','CO2','PMV']]
feature = ['Outsides_tem','Temperature','Related_humidity','CO2']
label = ['PMV']

x = data[feature]
x = preprocessing.StandardScaler().fit_transform(x)
y = data[label]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1)

model = Sequential()  # Sequential model
model.add(Dense(13, input_shape=(x_train.shape[1],)))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='mse',optimizer="adam")
history = model.fit(x_train,y_train,epochs=100,batch_size=256,
                    validation_data=(x_test,y_test))
loss = model.evaluate(x_test,y_test)

print(loss)

import matplotlib.pyplot as plt
#p = data[['PMV','PMV_pred1']].plot(subplots = True, style=['r-*','b-o'])

epochs=range(len(history.history['loss']))
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'y',label='Validation loss')
plt.title('Traing and Validation loss of first try 5')
plt.show()