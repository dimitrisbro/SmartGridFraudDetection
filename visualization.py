import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

rawData1 = pd.read_csv('visualization.csv', nrows=3)
cols = rawData1.columns
rawData2 = pd.read_csv('visualization.csv', skiprows=40254)
rawData2.columns = cols
data = pd.concat([rawData1, rawData2], ignore_index=True)

fig, axs = plt.subplots(2, 1)
fig.suptitle('Consumers With Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[0].plot(ax=axs[0], color='firebrick', grid=True)
axs[0].set_title('Consumer 0', fontsize=16)
axs[0].set_xlabel('Dates of Consumption')
axs[0].set_ylabel('Consumption')

data.loc[2].plot(ax=axs[1], color='firebrick', grid=True)
axs[1].set_title('Consumer 1', fontsize=16)
axs[1].set_xlabel('Dates of Consumption')
axs[1].set_ylabel('Consumption')

fig, axs = plt.subplots(2, 1)
fig.suptitle('Consumers Without Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[3].plot(ax=axs[0], color='teal', grid=True)
axs[0].set_title('Consumer 40255', fontsize=16)
axs[0].set_xlabel('Dates of Consumption')
axs[0].set_ylabel('Consumption')

data.loc[4].plot(ax=axs[1], color='teal', grid=True)
axs[1].set_title('Consumer 40256', fontsize=16)
axs[1].set_xlabel('Dates of Consumption')
axs[1].set_ylabel('Consumption')

fig2, axs2 = plt.subplots(2, 2)
fig2.suptitle('Statistics for Consumers without Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.8)

data.loc[0].plot(ax=axs2[0, 0], color='firebrick', grid=True)
axs2[0, 0].set_title('Consumption of Consumer 0', fontsize=16)
axs2[0, 0].set_xlabel('Dates of Consumption')
axs2[0, 0].set_ylabel('Consumption')

data.loc[0].hist(color='firebrick', ax=axs2[0, 1], grid=True)
axs2[0, 1].set_title('Histogram', fontsize=16)
axs2[0, 1].set_xlabel('Values')
axs2[0, 1].set_ylabel('Frequency')

data.loc[0].plot.kde(color='firebrick', ax=axs2[1, 0], grid=True)
axs2[1, 0].set_title('Density Estimation', fontsize=16)
axs2[1, 0].set_xlabel('Values')
axs2[1, 0].set_ylabel('Density')

data.loc[0].describe().drop(['count']).plot(kind='bar', ax=axs2[1, 1], color='firebrick', grid=True)
axs2[1, 1].set_title('Statistics', fontsize=16)
axs2[1, 1].set_ylabel('Values')

fig3, axs3 = plt.subplots(2, 2)
fig3.suptitle('Statistics for Consumers without Fraud', fontsize=18)
plt.subplots_adjust(hspace=0.8)
data.loc[4].plot(ax=axs3[0, 0], color='teal', grid=True)
axs3[0, 0].set_title('Consumption of Consumer 40256', fontsize=16)
axs3[0, 0].set_xlabel('Dates of Consumption')
axs3[0, 0].set_ylabel('Consumption')

data.loc[4].hist(color='teal', ax=axs3[0, 1])
axs3[0, 1].set_title('Histogram', fontsize=16)
axs3[0, 1].set_xlabel('Values')
axs3[0, 1].set_ylabel('Frequency')

data.loc[4].plot.kde(color='teal', ax=axs3[1, 0], grid=True)
axs3[1, 0].set_title('Density Estimation', fontsize=16)
axs3[1, 0].set_xlabel('Values')
axs3[1, 0].set_ylabel('Density')

data.loc[4].describe().drop(['count']).plot(kind='bar', ax=axs3[1, 1], color='teal', grid=True)
axs3[1, 1].set_title('Statistics', fontsize=16)
axs3[1, 1].set_ylabel('Values')

fig4, axs4 = plt.subplots(2, 1)
fig4.suptitle('Four Week Consumption', fontsize=16)
plt.subplots_adjust(hspace=0.5)

for i in range(59, 83, 7):
    axs4[0].plot(data.iloc[0, i:i + 7].to_numpy(), marker='>', linestyle='-', label='$week {i}$'.format(i=(i % 58) % 6))
axs4[0].legend(loc='best')
axs4[0].set_title('Without Fraud', fontsize=14)
axs4[0].set_ylabel('Consumption')
axs4[0].grid(True)

for i in range(59, 83, 7):
    axs4[1].plot(data.iloc[4, i:i + 7].to_numpy(), marker='>', linestyle='-', label='$week {i}$'.format(i=(i % 58) % 6))
axs4[1].legend(loc='best')
axs4[1].set_title('With Fraud', fontsize=14)
axs4[1].set_ylabel('Consumption')
axs4[1].grid(True)

fig5, axs5 = plt.subplots(1, 2)
a = []
for i in range(59, 83, 7):
    a.append(data.iloc[0, i:i + 7].to_numpy())
cor = pd.DataFrame(a).transpose().corr()
cax = axs5[0].matshow(cor)
for (i, j), z in np.ndenumerate(cor):
    axs5[0].text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')
alpha = ['week 1', 'week 2', 'week 3', 'week 4']
axs5[0].set_xticklabels([''] + alpha)
axs5[0].set_yticklabels([''] + alpha)
axs5[0].set_title('Customer without Fraud', fontsize=16)

a = []
for i in range(59, 83, 7):
    a.append(data.iloc[4, i:i + 7].to_numpy())
cor = pd.DataFrame(a).transpose().corr()
cax = axs5[1].matshow(cor)
for (i, j), z in np.ndenumerate(cor):
    axs5[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white')
axs5[1].set_xticklabels([''] + alpha)
axs5[1].set_yticklabels([''] + alpha)
axs5[1].set_title('Customer with Fraud', fontsize=16)
fig5.colorbar(cax)
# plt.close('all')
plt.show()
