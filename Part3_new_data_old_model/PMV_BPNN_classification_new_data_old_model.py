import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.layers.core import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

sourcefile = "../Concatenated_data/filtered_data.csv"
data = pd.read_csv(sourcefile)

feature = ['Outsides_tem','Temperature','Related_humidity','CO2']
label = ['PMV_category']

x = data[feature]
x = preprocessing.StandardScaler().fit_transform(x)
y = data[label]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1)

# split the training data and test data
y_train = np_utils.to_categorical(y_train,num_classes=7)
y_test = np_utils.to_categorical(y_test,num_classes=7)

#build the model
model = Sequential()
model.add(Dense(13,input_shape=(x_train.shape[1],)))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=50,
                    batch_size=256,validation_data=(x_test,y_test))

loss,accuracy = model.evaluate(x_test,y_test)
model.summary()

import matplotlib.pyplot as plt
epochs=range(len(history.history['loss']))
plt.plot(epochs,history.history['accuracy'],'b',label='Training loss')
plt.title('Accuracy of the model')
plt.show()

print(loss)
print(accuracy)