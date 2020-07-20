import pandas as pd
from sklearn.preprocessing import MinMaxScaler

rawData = pd.read_csv('data.csv')
pData = pd.DataFrame()
pData['FLAG'] = rawData['FLAG']
pData['CONS_NO'] = rawData['CONS_NO']
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

dIndex = data[data.duplicated()].index  # duplicates drop
data = data.drop(dIndex, axis=0)
pData = pData.drop(dIndex, axis=0)

zIndex = data[(data.sum(axis=1) == 0)].index  # zero rows drop
data = data.drop(zIndex, axis=0)
pData = pData.drop(zIndex, axis=0)

data.columns = pd.to_datetime(data.columns)  # columns reindexing according to dates
data = data.reindex(sorted(data.columns), axis=1)
cols = data.columns

data.reset_index(inplace=True, drop=True)  # index sorting
pData.reset_index(inplace=True, drop=True)

data = data.interpolate(method='linear', limit=2,  # filling NaN values
                        limit_direction='both', axis=0).fillna(0)

for i in range(data.shape[0]):  # outliers treatment
    m = data.loc[i].mean()
    st = data.loc[i].std()
    data.loc[i] = data.loc[i].mask(data.loc[i] > (m + 3 * st), other=m + 3 * st)

# data.to_csv(r'visualization.csv', index=False, header=True)

scale = MinMaxScaler()
# scale.fit(data)  # Scaling by date
# nData = pd.DataFrame(scale.transform(data))
# nData.columns = cols
# print(nData)
# preprData = pd.concat([pData, nData], axis=1, sort=False)  # Back to initial format
# print(preprData)
# # preprData.to_csv(r'preprocessedC.csv', index=False, header=True)

scaled_rows = scale.fit_transform(data.values.T).T
mData = pd.DataFrame(data=scaled_rows, columns=data.columns)
print(mData)
preprData = pd.concat([pData, mData], axis=1, sort=False)  # Back to initial format
print(preprData)
preprData.to_csv(r'preprocessedR.csv', index=False, header=True)