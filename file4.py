import pandas as pd
import numpy as np

rawData = pd.read_csv('/home/db/Documents/dipl/export_dataframe12.csv')
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

# print(data.skew(axis=1))

cols = data.columns
newData = pd.DataFrame(columns=cols)
index = data[abs(data.skew(axis=1)) > 2.5].index
for i in index:
    q = data.loc[i].quantile(0.8)
    data.loc[i] = data.loc[i].where(data.loc[i] < q, q)

# print(data[data.max(axis=1)>10*data.mean(axis=1)].index) #watch: 571 stoixeia poy exoume max > 10*mean
# print(data[data.max(axis=1)>20*data.mean(axis=1)].index) #watch:  0  stoixeia poy exoume max > 20*mean

#data.to_csv(r'/home/db/Documents/dipl/export_dataframe14.csv', index=False, header=True)

'''
cols=data.columns
newData=pd.DataFrame(columns=cols)
for i in range(32):
    q=data.loc[i].quantile(0.9)
    newData.loc[i] = data.loc[i].where(data.loc[i] < q, q)#todo check results
    if i%100==0:
        print(i)
#print(newData.skew(axis=1))


#newData.to_csv(r'/home/db/Documents/dipl/export_dataframe14.csv', index=False, header=True)
'''
