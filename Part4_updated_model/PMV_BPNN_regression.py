# -*- coding: utf-8 -*-
import pandas as pd
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,KFold,cross_validate
from sklearn.utils import shuffle
from keras.optimizers import RMSprop
from keras.layers.core import Dense,Dropout,Activation

#1 数据输入
input_file = "../Concatenated_data/filtered_data.csv"
data = pd.read_csv(input_file,encoding='gbk')
data = data[['Outsides_tem','Temperature','Related_humidity','CO2','PMV']]
# data = shuffle(data)
feature = ['Outsides_tem','Temperature','Related_humidity','CO2']
label = ['PMV']

x = data[feature]
x = preprocessing.StandardScaler().fit_transform(x)
y = data[label]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')
early_stop = EarlyStopping(monitor='loss', patience=5, mode='auto')
rmsprop = RMSprop(learning_rate=0.001)
callback_list = [reduce_lr,early_stop]

# 函数型模型
def create_model():
    model = Sequential()  #层次模型
    model.add(Dense(32,activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mse',optimizer="adam")
    return model

# build the model directly
# model = Sequential()  #层次模型
# model.add(Dense(32, activation='relu', input_shape=(x_train.shape[1],)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.compile(loss='mse',optimizer='adam')
# history = model.fit(x_train,y_train,epochs=100,batch_size=256,validation_data=(x_test,y_test),callbacks=reduce_lr)

# evaluate the model directly
# y_pred = model.predict(x_test)
# print("mse is "+str(mean_squared_error(y_test,y_pred)))
# print("rmse is " + str(mean_squared_error(y_test,y_pred)**0.5))
# print("mae is " + str(mean_absolute_error(y_test,y_pred)))
# print("r2 is " +str(r2_score(y_test,y_pred)))

# visualize the model
# import matplotlib.pyplot as plt
# # p = data[['PMV','y_pred']].plot(subplots = True, style=['r-*','b-o'])
#
# epochs=range(len(history.history['loss']))
# plt.plot(epochs,history.history['loss'],'b',label='Training loss')
# plt.plot(epochs,history.history['val_loss'],'y',label='Validation loss')
# plt.title('Traing and Validation loss 3')
# plt.show()

# 5-fold cross validation
estimator = KerasRegressor(build_fn=create_model,epochs=100,batch_size=256)
Kfold = KFold(n_splits=5,shuffle=True,random_state=1)
result = cross_validate(estimator,x,y,cv=Kfold,fit_params={'callbacks':callback_list},
                        scoring=('neg_mean_squared_error','neg_root_mean_squared_error',
                                 'neg_mean_absolute_error','r2'))

print(sorted(result.keys()))
print("mse is " + str(result['test_neg_mean_squared_error'].mean()*-1))
print("rmse is " + str(result['test_neg_root_mean_squared_error'].mean()*-1))
print("mae is " + str(result['test_neg_mean_absolute_error'].mean()*-1))
print("r2 is " + str(result['test_r2'].mean()))
print("fitting time is " + str(result['fit_time'].mean()))

