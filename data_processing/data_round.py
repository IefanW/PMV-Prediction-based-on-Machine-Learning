import pandas as pd

data = pd.read_csv("/Users/iefan_wey/Desktop/毕业设计/BPNN/Implementation/data.csv")

data["PMV_category"] = round(data["PMV"])
data.to_csv("/Users/iefan_wey/Desktop/毕业设计/BPNN/Implementation/filtered_data.csv",index=False)