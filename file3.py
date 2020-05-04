# statistika kai normalization

import pandas as pd
import matplotlib.pyplot as plt

#rawData = pd.read_csv('/home/db/Documents/dipl/preprocessed.csv')
data = pd.read_csv('/home/db/Documents/dipl/visualization.csv')
#data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

#print(data.loc[7362].mean())            #watch -->max peirazei ola ta ypoloipa xwris logo
#print(data['2014/10/10'].loc[7362])     #watch -->max peirazei ola ta ypoloipa xwris logo

totConsPerDay = data.sum(axis=0, skipna=True)
print(totConsPerDay)
totCons = totConsPerDay.sum()
print("Total Consumption:   ",totCons,"kWh")


totConsPerDay.plot(title="Total Consumption per Day",color='firebrick')
plt.show()

customer = pd.DataFrame()
customer['min'] = data.min(axis=1, skipna=True)
customer['max'] = data.max(axis=1, skipna=True)
customer['average'] = data.mean(axis=1, skipna=True)
customer['var'] = data.var(axis=1, skipna=True)
customer['totCons'] = data.sum(axis=1, skipna=True)
print("Total Consumer Statistics\n",customer.describe())


totMax = data.max().max()
totMin = data.min().min()
totMean = data.mean().mean()


print("Minimum Value:       ", totMin, "kWh")
print("Maximum Value:       ", totMax, "kWh")
print("Mean Value:          ", totMean, "kWh")

#print(data[data.max(axis=1)>10*data.mean(axis=1)].index)
#print(data[data.max(axis=1)>20*data.mean(axis=1)].index)
#print(data.max().sort_values())

