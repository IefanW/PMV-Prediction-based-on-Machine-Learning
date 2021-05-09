from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import tree
import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import joblib
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from keras.utils import np_utils

df = pd.read_csv("../Concatenated_data/filtered_data.csv")

feature_tag = ['Outsides_tem','Temperature','Related_humidity','CO2']
label_tag = ['PMV_category']

#feature and target data split
feature = df[feature_tag].values
PMV = df[label_tag].values.astype(str)

# traning and testing data split
feature_train, feature_test, label_train, label_test = train_test_split(feature, PMV, test_size=0.1,random_state=1)


# model building
classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(feature_train,label_train)
print(type(feature_train))
print(feature_train.shape)
print(feature_test.shape)

# prediction and evaluation
predict_results=classifier.predict(feature_test)
print(accuracy_score(predict_results, label_test))
conf_mat = multilabel_confusion_matrix(label_test, predict_results)
print(conf_mat)

joblib.dump(classifier,"shell_application/trained_model.pkl")

# cross validation
# estimator = classifier
# kfold = KFold(n_splits=5,shuffle=True,random_state=1)
# result = cross_validate(estimator,feature,PMV,cv=kfold,
#                          scoring=('accuracy','f1_micro','f1_macro'))
#
# print(sorted(result.keys()))
# print("accuracy is " + str(result['test_accuracy'].mean()))
# print("precision is " + str(result['test_f1_macro'].mean()))
# print("recall is " + str(result['test_f1_micro'].mean()))
# print("fitting time is " + str(result['fit_time'].mean()))