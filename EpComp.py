import pandas as pd
import matplotlib.pyplot as plt

nn = pd.read_csv('NN epochs results.csv')
nn = nn.set_index('Epochs')

# print(nn)

fig, axs = plt.subplots(2, 1)
fig.suptitle('Neural Network', fontsize=18)
plt.subplots_adjust(hspace=0.8)
nn[['AUC', 'Accuracy']].plot(ax=axs[0], grid=True)
axs[0].set_title('AUC and Accuracy dependance on Epochs ', fontsize=16)
axs[0].set_xlabel('Number of Epochs')

nn[['F1 (0)', 'F1 (1)']].plot(ax=axs[1], grid=True)
axs[1].set_title('F1 (0) and F1 (1) ', fontsize=16)
axs[1].set_xlabel('Number of Epochs')


cnn1 = pd.read_csv('CNN epochs results.csv')
cnn1 = nn.rename(columns={"NN": "Epochs", "accuracy": "Accuracy"}).set_index('Epochs')

# print(nn)

fig2, axs2 = plt.subplots(2, 1)
fig2.suptitle('1D - Convolutional Neural Network', fontsize=18)
plt.subplots_adjust(hspace=0.8)
cnn1[['AUC', 'Accuracy']].plot(ax=axs2[0], grid=True)
axs2[0].set_title('AUC and Accuracy dependance on Epochs ', fontsize=16)
axs2[0].set_xlabel('Number of Epochs')

cnn1[['F1 (0)', 'F1 (1)']].plot(ax=axs2[1], grid=True)
axs2[1].set_title('F1 (0) and F1 (1) ', fontsize=16)
axs2[1].set_xlabel('Number of Epochs')

cnn2 = pd.read_csv('CNN epochs results.csv')
cnn2 = nn.rename(columns={"NN": "Epochs", "accuracy": "Accuracy"}).set_index('Epochs')

# print(nn)

fig3, axs3 = plt.subplots(2, 1)
fig.suptitle('2D - Convolutional Neural Network', fontsize=18)
plt.subplots_adjust(hspace=0.8)
cnn2[['AUC', 'Accuracy']].plot(ax=axs3[0], grid=True)
axs3[0].set_title('AUC and Accuracy dependance on Epochs ', fontsize=16)
axs3[0].set_xlabel('Number of Epochs')

cnn2[['F1 (0)', 'F1 (1)']].plot(ax=axs3[1], grid=True)
axs3[1].set_title('F1 (0) and F1 (1) ', fontsize=16)
axs3[1].set_xlabel('Number of Epochs')
plt.show()
