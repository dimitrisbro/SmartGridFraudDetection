# cleaning up from outliers

import pandas as pd

rawData = pd.read_csv('/home/db/Documents/dipl/export_dataframe12.csv')
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)
data.columns = pd.to_datetime(data.columns)
data = data.reindex(sorted(data.columns), axis=1)

#print(2*data.loc[1].std()+data.loc[1].mean())

#todo symplhrwsh nan grammika se 2 theseis kai ta ypoloipa me 0
print('Total NaN:                       ',data.isnull().sum().sum())
print('Total NaN after Interpolation:   ',data.interpolate(method='linear',limit=2,
                      limit_direction='both',axis=0).isnull().sum().sum())
#data=data.interpolate(method='linear',limit=2,
#                      limit_direction='both',axis=0).fillna(0)
'''
#todo mask gia outliers

for i in range(5):
    m=data.loc[i].mean()
    st=data.loc[i].std()
    data.loc[i]=data.loc[i].mask(data.loc[i]>(m+2*st),other=m+2*st)
print(data.max(axis=1).head())


# print(data.skew(axis=1))
'''
'''
cols = data.columns
index = data[abs(data.skew(axis=1)) > 2.5].index
for i in index:
    q = data.loc[i].quantile(0.8)
    #if q != 0:
    data.loc[i] = data.loc[i].where(data.loc[i] < q, q)
    '''
'''
nData = pd.DataFrame()
nData['CONS_NO'] = rawData['CONS_NO']
nData['FLAG'] = rawData['FLAG']

newData = pd.concat([nData, data], axis=1, sort=False)
'''
#print(data[data.max(axis=1)>10*data.mean(axis=1)].index) #watch: 571 stoixeia poy exoume max > 10*mean
#print(data[data.max(axis=1)>20*data.mean(axis=1)].index) #watch:  0  stoixeia poy exoume max > 20*mean
#print(data[data.max(axis=1)==0].index)  #todo polla mhdenika

# newData.to_csv(r'/home/db/Documents/dipl/export_dataframe14.csv', index=False, header=True)

'''
cols=data.columns
newData=pd.DataFrame(columns=cols)
for i in range(32):
    q=data.loc[i].quantile(0.9)
    newData.loc[i] = data.loc[i].where(data.loc[i] < q, q)#todo check results
    if i%100==0:
        print(i)
#print(newData.skew(axis=1))
'''
