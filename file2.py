# afairesh mhdenikwn grammwn apo to dataset

import pandas as pd

rawData = pd.read_csv('/home/db/Documents/dipl/export_dataframe1.csv')
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

index = data[(data.max(axis=1) == 0)].index
print('Raws with Zero Consumption\n',index)
newData = rawData.drop(index, axis=0)
print(newData)
# newData.to_csv(r'/home/db/Documents/dipl/export_dataframe12.csv', index=False, header=True)


"""
l = 0
flag = 1
for i in data.max(axis=1):
    if i == 0:
        if flag == 1:
            dataNew = rawData.drop([l], axis=0)
            flag = 0
        else:
            dataNew = dataNew.drop([l], axis=0)
    #print(l)
    l = l + 1

#dataNew.to_csv(r'/home/db/Documents/dipl/ElectricityTheftDetection/data/export_dataframe12.csv', index=False, header=True)
print(dataNew)
"""
