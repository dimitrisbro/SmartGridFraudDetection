import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Conv2D
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
import rbm
import rbmD
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

tf.random.set_seed(1234)


def roc(pred, y_test):
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], #color='darkorange'
             ''',lw=lw ,label='ROC curve (area = %0.2f)' % roc_auc[2]''')
    plt.plot([0, 1], [0, 1])#, color='navy', lw=lw, linestyle='--'
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    plt.show()


def NN(X_train, X_test, y_train, y_test):
    print('Neural Network:')
    # for i in range(4,100,3):
    #     print("Epoch:",i)
    model = Sequential()
    model.add(Dense(1000, input_dim=1034, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(trainX, y_train, validation_split=0, epochs=i, shuffle=True, verbose=0)
    model.fit(X_train, y_train, validation_split=0, epochs=4, shuffle=True, verbose=1)
    prediction = model.predict_classes(X_test)
    # prediction = np.argmax(model.predict(X_test), axis=-1)
    print("ACC", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")
    # todo :roc(prediction,y_test)
    # acc_per_fold = []
    # loss_per_fold = []
    # kfold = KFold(n_splits=10, shuffle=True)
    # f = 1
    # for train, test in kfold.split(X, y):
    #     print("Fold Number %s" % (f))
    #     model = Sequential()
    #     model.add(Dense(1000, input_dim=1034, activation='relu'))
    #     model.add(Dense(100, activation='relu'))
    #     model.add(Dense(100, activation='relu'))
    #     model.add(Dense(100, activation='relu'))
    #     model.add(Dense(10, activation='relu'))
    #     model.add(Dense(1, activation='sigmoid'))
    #
    #     model.compile(loss=keras.losses.binary_crossentropy,
    #                   optimizer='adam',
    #                   metrics=['accuracy'])
    #
    #     model.fit(X.iloc[train], y.iloc[train], validation_split=0.1, epochs=4, shuffle=True)
    #     scores = model.evaluate(X.iloc[test], y.iloc[test], verbose=0)
    #     acc_per_fold.append(scores[1] * 100)
    #     loss_per_fold.append(scores[0])
    #     f = f + 1
    # print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    # print(f'> Loss: {np.mean(loss_per_fold)}')


def CNN1D(X_train, X_test, y_train, y_test):
    print('1D - Convolutional Neural Network:')
    # acc_per_fold = []
    # loss_per_fold = []
    # au = []
    # kfold = KFold(n_splits=10, shuffle=True)
    # f = 1
    # for train, test in kfold.split(X, y):
    #     print("Fold Number %s" % (f))
    # todo: cross validation

    #     trainX = trainX.to_numpy().reshape(X.iloc[train].shape[0], X.iloc[train].shape[1], 1)
    #     X_test = X_test.to_numpy().reshape(X.iloc[test].shape[0], X.iloc[test].shape[1], 1)
    #     model = Sequential()
    #     model.add(Conv1D(100, kernel_size=7, input_shape=(1034, 1), activation='relu'))
    #     # model.add(Flatten())
    #     # model.add(Conv1D(32, kernel_size=(7), activation='relu'))
    #     # model.add(Dropout(0.5))
    #     # model.add(MaxPooling1D(pool_size=2))
    #     # model.add(Flatten())
    #     # model.add(Dense(100, activation='relu'))
    #     # model.add(Dense(100, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(1, activation='sigmoid'))
    #
    #     model.compile(loss=keras.losses.binary_crossentropy,
    #                   optimizer='adam',
    #                   metrics=['accuracy'])
    #
    #     # model.summary()
    #
    #     model.fit(trainX, y_train, epochs=2, validation_split=0.1, shuffle=False)
    #     au.append(roc_auc_score(model.predict_classes(X_test),y_test))
    #     scores = model.evaluate(trainX, y_test, verbose=0)
    #     acc_per_fold.append(scores[1] * 100)
    #     loss_per_fold.append(scores[0])
    #     f = f + 1
    # print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    # print(f'> AUC: {np.mean(au)} (+- {np.std(au)})')
    # print(f'> Loss: {np.mean(loss_per_fold)}')

    X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

    # for i in range(4,100,3):
    #     print("Epoch:",i)
    model = Sequential()
    model.add(Conv1D(100, kernel_size=7, input_shape=(1034, 1), activation='relu'))
    # model.add(Flatten())
    # model.add(Conv1D(32, kernel_size=(7), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.summary()

    # model.fit(trainX, y_train, epochs=1, validation_split=0.1, shuffle=False, verbose=1)
    model.fit(X_train, y_train, epochs=4, validation_split=0.1, shuffle=False, verbose=1)
    prediction = model.predict_classes(X_test)
    print("ACC", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def CNN2D(X_train, X_test, y_train, y_test):
    print('2D - Convolutional Neural Network:')
    nX = X_train.to_numpy()  # trainX to 2D - array
    b = np.hstack((nX, np.zeros((nX.shape[0], 2))))
    # print(b.shape[0], "-")
    l = []
    for i in range(b.shape[0]):
        a = np.reshape(b[i], (-1, 7, 1))
        l.append(a)
    m = np.array(l)
    # print(m.shape, "-")

    nXt = X_test.to_numpy()  # X_test to 2D - array
    d = np.hstack((nXt, np.zeros((nXt.shape[0], 2))))
    # print(d.shape[0])
    k = []
    for i in range(d.shape[0]):
        c = np.reshape(d[i], (-1, 7, 1))
        k.append(c)
    n = np.array(k)
    # print(n.shape)
    input_shape = (1, 148, 7, 1)

    # for i in range(4,100,3):
    #     print("Epoch:",i)
    model = Sequential()
    model.add(Conv2D(kernel_size=(7, 3), filters=32, input_shape=input_shape[1:], activation='relu',
                     data_format='channels_last'))
    model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.summary()
    #     model.fit(m, y_train, validation_split=0.1, epochs=i, shuffle=False, verbose=0)
    model.fit(m, y_train, validation_split=0.1, epochs=1, shuffle=False, verbose=1)

    # prediction = model.predict_classes(X_test)
    prediction = model.predict_classes(n)
    # print(prediction)
    print("ACC", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def LR(X_train, X_test, y_train, y_test):
    print('Logistic Regression:')
    # param_grid = {'C': [0.1,1,10,100],'solver': ['newton-cg', 'lbfgs']}
    # grid = GridSearchCV(LogisticRegression(max_iter=1000,random_state=0), param_grid=param_grid, n_jobs=-1)
    # grid.fit(trainX, y_train)
    # df = pd.DataFrame(grid.cv_results_)
    # print(df[['param_C', 'param_solver', 'mean_test_score', 'rank_test_score']])

    model = LogisticRegression(C=1000, max_iter=1000, n_jobs=-1, solver='newton-cg')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("ACC", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def DT(X_train, X_test, y_train, y_test):
    print('Decision Tree:')
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("ACC", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def RF(X_train, X_test, y_train, y_test):
    print('Random Forest:')
    # param_grid = {'n_estimators':[10,100,1000]}
    # grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid=param_grid, n_jobs=-1)
    # grid.fit(trainX, y_train)
    # df = pd.DataFrame(grid.cv_results_)
    # print(df[['param_criterion', 'mean_test_score', 'rank_test_score']])

    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='auto',  # max_depth=10,
                                   random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("ACC:", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def SVM(X_train, X_test, y_train, y_test):
    model = SVC(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('SVM:')
    print("ACC:", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


def RBM(X_train, X_test, y_train, y_test):
    print("sklearn RBM + NN")
    # logistic = LogisticRegression(solver='newton-cg', tol=1)
    nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 25, 1), random_state=1)
    rbm = BernoulliRBM(random_state=0, verbose=True)

    rbm_features_classifier = Pipeline(
        # steps=[('rbm', rbm), ('logistic', logistic)])
        steps=[('rbm', rbm), ('nn', nn)])
    rbm.learning_rate = 0.06
    rbm.n_iter = 2

    rbm.n_components = 100
    # logistic.C = 6000

    rbm_features_classifier.fit(X_train, y_train)
    prediction = rbm_features_classifier.predict(X_test)
    print("ACC:", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")
    # print("Logistic regression using RBM features:\n%s\n" % (classification_report(y_test, prediction)))


def RBM1(X_train, X_test, y_train, y_test):
    print("Custom RBM:")
    n = X_train.to_numpy()
    g1 = rbm.RBM(num_visible=1034, num_hidden=1)
    g1.train(n,max_epochs=4)
    user = X_test.to_numpy()
    # print(np.array([trainX.loc[2732].to_numpy()]))
    prediction = pd.DataFrame(g1.run_visible(user))
    # print(prediction)
    print("ACC:", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")
    # g2=rbm.RBM(num_visible = 500, num_hidden = 500)
    # g2.train(n)


def RBMD(X_train, X_test, y_train, y_test):
    print("Custom RBM")
    n = X_train.to_numpy()
    g1 = rbmD.RBM((1034,1000))
    g1.train(n, max_epochs=1
             )
    user = X_test.to_numpy()
    # print(np.array([trainX.loc[2732].to_numpy()]))
    inp = pd.DataFrame(g1.run_visible(n))
    prediction = pd.DataFrame(g1.run_visible(user))
    # print("ACC:", 100 * accuracy_score(y_test, prediction))
    # print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    # print("MAE:", mean_absolute_error(y_test, prediction))
    # print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    # print("ROC:", 100 * roc_auc_score(y_test, prediction))
    # print(confusion_matrix(y_test, prediction), "\n")
    print("Custom RBM + LR:")
    mod = LogisticRegression(C=10)
    mod.fit(inp, y_train)
    pl = mod.predict(prediction)
    # print(classification_report(y_test, pl))
    print(100 * accuracy_score(y_test, pl))
    print(confusion_matrix(y_test, pl), "\n")
    print("ACC:",100 * accuracy_score(y_test, pl))
    print("RMSE:", mean_squared_error(y_test, pl,squared=False))
    print("MAE:", mean_absolute_error(y_test, pl))
    print("F1:", 100 * precision_recall_fscore_support(y_test, pl)[2])
    print("ROC:", 100 * roc_auc_score(y_test, pl))
    print(confusion_matrix(y_test, pl), "\n")
    # g2=rbm.RBM(num_visible = 500, num_hidden = 500)
    # g2.train(n)


def LRBM(X_train, X_test, y_train, y_test):
    print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
    logistic = LogisticRegression(C=1.0, max_iter=1000)
    logistic.fit(X_train, y_train)
    print("LOGISTIC REGRESSION ON ORIGINAL DATASET")
    # print(classification_report(y_test, logistic.predict(X_test)))
    # initialize the RBM + Logistic Regression classifier with
    # the cross-validated parameters
    rbm1 = BernoulliRBM(n_components=500, n_iter=3,
                        learning_rate=0.01, verbose=True)
    rbm2 = BernoulliRBM(n_components=1, n_iter=3,
                        learning_rate=0.01, verbose=True)
    logistic = LogisticRegression(C=1.0)
    # train the classifier and show an evaluation report
    classifier = Pipeline([("rbm1", rbm1), ("rbm2", rbm2), ("logistic", logistic)])
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    # print(classification_report(y_test, prediction))
    print("ACC:", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("ROC:", 100 * roc_auc_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction), "\n")


rawData = pd.read_csv('preprocessedR.csv')
# X= pd.read_csv('noisePoisson.csv')
y = rawData[['FLAG']]
X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

''''''
print('Normal Consumers:                    ', y[y['FLAG'] == 0].count()[0])
print('Total Consumers:                     ', y.shape[0])
print("Classification assuming no fraud:     %.2f" % (y[y['FLAG'] == 0].count()[0] / y.shape[0] * 100), "%")

X.columns = pd.to_datetime(X.columns)  # columns reindexing according to dates
X = X.reindex(X.columns, axis=1)
# print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y['FLAG'], test_size=0.1, random_state=0)
print(X_train.shape, y_train.shape, y_train[y_train == 1].count(), y_train[y_train == 0].count())
print(X_test.shape, y_test.shape, y_test[y_test == 1].count(), y_test[y_test == 0].count())

# over = SMOTE(sampling_strategy=0.2, random_state=0)
# trainX,y_train=over.fit_resample(trainX,y_train)
# print(trainX.shape,y_train.shape,y_train[y_train==1].count(),y_train[y_train==0].count())
print("Test set assuming no fraud:     %.2f" % (y_test[y_test == 0].count() / y_test.shape[0] * 100), "%")

print(X.head())

# NN(trainX, X_test, y_train, y_test)
# CNN1D(trainX, X_test, y_train, y_test)
# CNN2D(trainX, X_test, y_train, y_test)
# SVM(trainX, X_test, y_train, y_test)
# RBM(trainX, X_test, y_train, y_test)
# RBM1(trainX, X_test, y_train, y_test)
RBMD(X_train, X_test, y_train, y_test)
# LRBM(trainX, X_test, y_train, y_test)
# RF(trainX, X_test, y_train, y_test)
# LR(trainX, X_test, y_train, y_test)
# DT(trainX, X_test, y_train, y_test)
