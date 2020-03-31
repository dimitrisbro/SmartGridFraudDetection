# afairesh duplicates apo to dataset

import pandas as pd

rawData = pd.read_csv('/home/db/Documents/dipl/data.csv')
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

dupData = data.duplicated()
y = data[dupData].index
newData = rawData.drop(y, axis=0)
print(newData)
# newData.to_csv(r'/home/db/Documents/diplexport_dataframe1.csv', index=False, header=True)


'''
flag = 1
for row in y.index:
    print(row)
    if flag == 1:
        newData = rawData.drop([row], axis=0)
        flag = 0
    else:
        newData = newData.drop([row], axis=0)



#newData.to_csv(r'/home/db/Documents/dipl/ElectricityTheftDetection/data/export_dataframe1.csv', index=False, header=True)
print(newData)
'''
