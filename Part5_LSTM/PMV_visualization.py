import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_train = pd.read_csv("LSTM_data/data_train.csv")
data = dataset_train[['PMV']].values
data = data[:240]

plt.plot(data, color = 'red', label = 'PMV')
plt.title('PMV Period')
plt.xlabel('Number of data')
plt.ylabel('PMV')
plt.legend()
plt.show()