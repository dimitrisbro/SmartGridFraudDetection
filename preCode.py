import pandas as pd
from sklearn.preprocessing import MinMaxScaler

rawData = pd.read_csv('data.csv')
infoData = pd.DataFrame()
infoData['FLAG'] = rawData['FLAG']
infoData['CONS_NO'] = rawData['CONS_NO']
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

dropIndex = data[data.duplicated()].index  # duplicates drop
data = data.drop(dropIndex, axis=0)
infoData = infoData.drop(dropIndex, axis=0)

zeroIndex = data[(data.sum(axis=1) == 0)].index  # zero rows drop
data = data.drop(zeroIndex, axis=0)
infoData = infoData.drop(zeroIndex, axis=0)

data.columns = pd.to_datetime(data.columns)  # columns reindexing according to dates
data = data.reindex(sorted(data.columns), axis=1)
cols = data.columns

data.reset_index(inplace=True, drop=True)  # index sorting
infoData.reset_index(inplace=True, drop=True)

data = data.interpolate(method='linear', limit=2,  # filling NaN values
                        limit_direction='both', axis=0).fillna(0)

for i in range(data.shape[0]):  # outliers treatment
    m = data.loc[i].mean()
    st = data.loc[i].std()
    data.loc[i] = data.loc[i].mask(data.loc[i] > (m + 3 * st), other=m + 3 * st)

data.to_csv(r'visualization.csv', index=False, header=True)  # preprocessed data without scaling

scale = MinMaxScaler()
scaled = scale.fit_transform(data.values.T).T
mData = pd.DataFrame(data=scaled, columns=data.columns)
print(mData)
preprData = pd.concat([infoData, mData], axis=1, sort=False)  # Back to initial format
print(preprData)
preprData.to_csv(r'preprocessedR.csv', index=False, header=True)
