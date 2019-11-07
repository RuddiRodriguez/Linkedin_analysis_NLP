import pandas as pd
from pandas import DataFrame

list_names = (
    'demodata_training_0.csv', 'demodata_training_1.csv', 'demodata_training_2.csv', 'demodata_training_3.csv',
    'demodata_training_4.csv', 'demodata_training_5.csv', 'demodata_training_6.csv', 'demodata_training_7.csv',
    'demodata_training_8.csv', 'demodata_training_9.csv', 'demodata_training_10.csv', 'demodata_training_11.csv',
    'demodata_training_12.csv'
)

final_data: DataFrame = pd.DataFrame()

for items in list_names:
    print(items)
    data_temp = pd.read_csv(
        "/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/" + items)
    final_data = final_data.append(data_temp)

print(final_data.shape)
final_data.to_csv(
    r'/Users/ruddirodriguez/Dropbox/Machine_Learning/NLP/demodata_training_full_v1.csv ')
