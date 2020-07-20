import pandas as pd
import matplotlib.pyplot as plt

nn = pd.read_csv('NN epochs results.csv')
nn = nn.set_index('Epochs')

# print(nn)

fig, axs = plt.subplots(2, 1)
fig.suptitle('Neural Network', fontsize=18)
plt.subplots_adjust(hspace=0.8)
nn[['AUC', 'Accuracy']].plot(ax=axs[0], grid=True)
axs[0].set_title('AUC and Accuracy dependency on Epochs ', fontsize=16)
axs[0].set_xlabel('Number of Epochs')

nn[['F1 (0)', 'F1 (1)']].plot(ax=axs[1], grid=True)
axs[1].set_title('F1 (0) and F1 (1) ', fontsize=16)
axs[1].set_xlabel('Number of Epochs')

cnn1 = pd.read_csv('1D-Cnn Epochs Results.csv')
cnn1 = cnn1.set_index('Epochs')

# print(nn)

fig2, axs2 = plt.subplots(2, 1)
fig2.suptitle('1D - CNN', fontsize=18)
plt.subplots_adjust(hspace=0.8)
cnn1[['AUC', 'Accuracy']].plot(ax=axs2[0], grid=True)
axs2[0].set_title('AUC and Accuracy dependency on Epochs ', fontsize=14)
axs2[0].set_xlabel('Number of Epochs')

cnn1[['F1 (0)', 'F1 (1)']].plot(ax=axs2[1], grid=True)
axs2[1].set_title('F1 (0) and F1 (1) ', fontsize=16)
axs2[1].set_xlabel('Number of Epochs')

cnn2 = pd.read_csv('2D-CNN Epochs Results.csv')
cnn2 = cnn2.set_index('Epochs')

# print(nn)

fig3, axs3 = plt.subplots(2, 1)
fig3.suptitle('2D - CNN', fontsize=18)
plt.subplots_adjust(hspace=0.8)
cnn2[['AUC', 'Accuracy']].plot(ax=axs3[0], grid=True)
axs3[0].set_title('AUC and Accuracy dependency on Epochs ', fontsize=14)
axs3[0].set_xlabel('Number of Epochs')

cnn2[['F1 (0)', 'F1 (1)']].plot(ax=axs3[1], grid=True)
axs3[1].set_title('F1 (0) and F1 (1) ', fontsize=16)
axs3[1].set_xlabel('Number of Epochs')

fig4, axs4 = plt.subplots(2, 1)
fig4.suptitle('Models Comparison', fontsize=18)
plt.subplots_adjust(hspace=0.8)
df = pd.DataFrame()
df['NN'] = nn['AUC']
df['1D-CNN'] = cnn1['AUC']
df['2D-CNN'] = cnn2['AUC']
df[['NN', '1D-CNN', '2D-CNN']].plot(ax=axs4[0], grid=True)
axs4[0].set_title('AUC dependency on Epochs ', fontsize=14)
axs4[0].set_xlabel('Number of Epochs')

df = pd.DataFrame()
df['NN'] = nn['Accuracy']
df['1D-CNN'] = cnn1['Accuracy']
df['2D-CNN'] = cnn2['Accuracy']
df[['NN', '1D-CNN', '2D-CNN']].plot(ax=axs4[1], grid=True)
axs4[1].set_title('Accuracy dependency on Epochs ', fontsize=14)
axs4[1].set_xlabel('Number of Epochs')
plt.show()
