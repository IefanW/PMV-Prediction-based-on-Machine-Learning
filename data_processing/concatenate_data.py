import pandas as pd
import os

dir = '/Users/iefan_wey/Desktop/毕业设计/BPNN/Implementation/数据'

file_name_list = []
frames = []

for root, dirs, files in os.walk(dir):
    for csv_file in files:
        if csv_file.endswith('.csv'):
            file_name_list.append(os.path.join(root, csv_file))
            df = pd.read_csv(os.path.join(root, csv_file),encoding = 'gbk')
            print(df.columns.values)
            df.rename(columns={'日期/时间':'Datetime','Date/Time':'Datetime','日期-时间':'Datetime',

                               '环境:场地室外空气干球温度[C](小时)':'Outsides_tem', '环境: 现场室外空气干球温度 [C](Hourly)':'Outsides_tem',
                               '环境:场地室外空气干球温度 [C](小时)':'Outsides_tem', '环境:工地室外空气干球温度[C](小时)':'Outsides_tem',
                               'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)':'Outsides_tem',

                               '热区:空间102:区平均气温[C](小时)':'Temperature','热区:空间102:区平均气温[C](Hourly)':'Temperature',
                               '热区: 空间102: 区域平均气温 [C](Hourly)':'Temperature','热区: 空间102:区平均气温[C](小时)':'Temperature',
                               '热区:第102空间:区平均气温[C](每小时)':'Temperature', 'THERMAL ZONE: SPACE 102:Zone Mean Air Temperature [C](Hourly)':'Temperature',

                               '热区:空间102:区域空气相对湿度[%](小时)':'Related_humidity','热区:空间102:区域空气相对湿度[%](Hourly)':'Related_humidity',
                               '热区: 空间102: 区域空气相对湿度[%](Hourly)':'Related_humidity','热区: 空间102:区域空气相对湿度[%](小时)':'Related_humidity',
                               'THERMAL ZONE: SPACE 102:Zone Air Relative Humidity [%](Hourly)':'Related_humidity',

                               '热区:空间102:区空气CO2浓度[ppm](小时)':'CO2','热区: 空间102: 区域空气二氧化碳浓度 [ppm](Hourly)':'CO2',
                               'THERMAL ZONE: SPACE 102:Zone Air CO2 Concentration [ppm](Hourly)':'CO2',

                                '人:区域热舒适Fanger模型PMV[](小时)':'PMV','区域热舒适吸管模型 pmv [](Hourly)':'PMV',
                               'PEOPLE:Zone Thermal Comfort Fanger Model PMV [](Hourly)':'PMV'},
                      inplace=True)
            data = df[['Datetime','Outsides_tem','Temperature','Related_humidity','CO2','PMV']]
            frames.append(data)

result = data
print(result.head())
print(result.shape)
result.to_csv(result.to_csv('/Users/iefan_wey/Desktop/毕业设计/BPNN/Implementation/data.csv',index = False))