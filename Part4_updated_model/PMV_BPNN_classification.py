import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate, KFold, train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Activation, Dropout

sourcefile = "../Concatenated_data/filtered_data.csv"
data = pd.read_csv(sourcefile)

feature = ['Outsides_tem','Temperature','Related_humidity','CO2']
label = ['PMV_category']

shuffle(data)
x = data[feature]
x = preprocessing.StandardScaler().fit_transform(x)
y = data[label]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1)

# split the training data and test data
y_train = np_utils.to_categorical(y_train,num_classes=7)
y_test = np_utils.to_categorical(y_test,num_classes=7)

reduce_lr = ReduceLROnPlateau(monitor='accuracy', patience=10, mode='auto')
rmsprop = RMSprop(learning_rate=0.01)

#build the model
def create_model():
    model = Sequential()
    model.add(Dense(32,activation='relu',input_shape=(x_train.shape[1],)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(7,activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    # history = model.fit(x_train,y_train,epochs=10,
    #                 batch_size=256,validation_data=(x_test,y_test),
    #                 callbacks=reduce_lr)
    return model

# model = Sequential()
# model.add(Dense(32,activation='relu',input_shape=(x_train.shape[1],)))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(7,activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
# history = model.fit(x_train,y_train,epochs=100,
#                 batch_size=256,validation_data=(x_test,y_test),
#                 callbacks=reduce_lr)
# loss,accuracy = model.evaluate(x_test,y_test)

# visualize the model
# import matplotlib.pyplot as plt
# epochs=range(len(history.history['loss']))
# plt.plot(epochs,history.history['accuracy'],'b',label='Training loss')
# plt.title('Accuracy of the model-3')
# plt.show()
#
# print(loss)
# print(accuracy)

# cross validation
estimator = KerasClassifier(build_fn=create_model, epochs=1, batch_size=256)
kfold = KFold(n_splits=2,shuffle=True,random_state=1)
result = cross_validate(estimator,x,y,cv=kfold,
                         fit_params={'callbacks':reduce_lr},
                         scoring=('accuracy','f1_micro','f1_macro'))

print(sorted(result.keys()))
print("accuracy is " + str(result['test_accuracy'].mean()))
print("macro f1-score is " + str(result['test_f1_macro'].mean()))
print("micro f1-score is " + str(result['test_f1_micro'].mean()))
print("fitting time is " + str(result['fit_time'].mean()))

