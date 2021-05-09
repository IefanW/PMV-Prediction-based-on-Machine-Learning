from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, KFold, train_test_split
start = time.time()

df = pd.read_csv("../Concatenated_data/filtered_data.csv")
df = shuffle(df)
df['PMV_category'] = df['PMV_category'].astype('str')

# feature and target data split
feature = df.iloc[:,1:5]
PMV = df.iloc[:,-1]

# training and testing data split
feature_train, feature_test, label_train, label_test = train_test_split(feature, PMV, test_size=0.1,random_state=1)

# model building
classifier = RandomForestClassifier(criterion='gini')
classifier.fit(feature_train,label_train)

# prediction and evaluation
# predict_results=classifier.predict(feature_test)
# print(accuracy_score(predict_results, label_test))
# conf_mat = confusion_matrix(label_test, predict_results)
# print(conf_mat)
# print(classification_report(label_test, predict_results))

# cross validation
estimator = classifier
kfold = KFold(n_splits=5,shuffle=True,random_state=1)
result = cross_validate(estimator,feature,PMV,cv=kfold,
                         scoring=('accuracy','f1_micro','f1_macro'))

print(sorted(result.keys()))
print("accuracy is " + str(result['test_accuracy'].mean()))
print("precision is " + str(result['test_f1_macro'].mean()))
print("recall is " + str(result['test_f1_micro'].mean()))
print("fitting time is " + str(result['fit_time'].mean()))
