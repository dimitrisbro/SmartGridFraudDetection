# statistika kai normalization

import pandas as pd
import matplotlib.pyplot as plt

rawData = pd.read_csv('/home/db/Documents/dipl/export_dataframe12.csv')
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

# print(data['2014/10/10'].loc[7362])  # -->max peirazei ola ta ypoloipa xwris logo

totConsPerDay = data.sum(axis=0, skipna=True)
# print(totConsPerDay)
totCons = totConsPerDay.sum()
# print(totCons)

totConsPerDay.plot(title="Total Consumption per Day")
plt.show()

customer = pd.DataFrame()
customer['min'] = data.min(axis=1, skipna=True)
customer['max'] = data.max(axis=1, skipna=True)
customer['average'] = data.mean(axis=1, skipna=True)
customer['var'] = data.var(axis=1, skipna=True)
customer['totCons'] = data.sum(axis=1, skipna=True)

totMax = data.max().max()
totMin = data.min().min()
totMean = data.mean().mean()

'''
print(totMin)
print(totMax)
print(totMean)
'''

nData = pd.DataFrame()
nData['FLAG'] = rawData['FLAG']
nData['CONS_NO'] = rawData['CONS_NO']

normalized = (data - totMin) / (totMax - totMin)
normalizedData = pd.concat([nData, normalized], axis=1, sort=False)

print(normalizedData.head())

