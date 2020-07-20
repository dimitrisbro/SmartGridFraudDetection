import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
import rbm
import rbmD
import numpy as np

def NN(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(100, input_dim=1034, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0.1)
    prediction = model.predict_classes(X_test)
    print('Neural Network:')
    print(100 * accuracy_score(y_test, prediction))
    print(mean_squared_error(y_test, prediction,squared=True))
    print(mean_absolute_error(y_test, prediction))
    print(confusion_matrix(y_test, prediction))


def LR(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('Logistic Regression:')
    print(100 * accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))


def DT(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('Decision Tree:')
    print(100 * accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))


def RF(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('Random Forest:')
    print(100 * accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))


def SVM(X_train, X_test, y_train, y_test):
    model = SVC(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print('SVM:')
    print(100 * accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

def RBM(X_train, X_test, y_train, y_test):
    #logistic = LogisticRegression(solver='newton-cg', tol=1)
    nn=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(50, 25,1), random_state=1)
    rbm = BernoulliRBM(random_state=0, verbose=True)

    rbm_features_classifier = Pipeline(
        #steps=[('rbm', rbm), ('logistic', logistic)])
        steps=[('rbm', rbm), ('nn', nn)])
    rbm.learning_rate = 0.06
    rbm.n_iter = 10

    rbm.n_components = 100
    #logistic.C = 6000

    rbm_features_classifier.fit(X_train, y_train)
    prediction = rbm_features_classifier.predict(X_test)
    print(100 * accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    #print("Logistic regression using RBM features:\n%s\n" % (classification_report(y_test, prediction)))


def RBM1(X_train, X_test, y_train, y_test):
    n=X_train.to_numpy()
    g1=rbm.RBM(num_visible = 1034, num_hidden = 500)
    g1.train(n)
    user = X_test.to_numpy()
    #print(np.array([X_train.loc[2732].to_numpy()]))
    prediction=pd.DataFrame(g1.run_visible(user))
    #print(prediction)
    #print(100 * accuracy_score(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    #g2=rbm.RBM(num_visible = 500, num_hidden = 500)
    #g2.train(n)

def RBMD(X_train, X_test, y_train, y_test):
    n = X_train.to_numpy()
    g1 = rbmD.RBM((1000,800,1))
    g1.train(n,max_epochs=3)
    user = X_test.to_numpy()
    # print(np.array([X_train.loc[2732].to_numpy()]))
    prediction = pd.DataFrame(g1.run_visible(user))
    #print(prediction)
    print(100 * accuracy_score(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    # g2=rbm.RBM(num_visible = 500, num_hidden = 500)
    # g2.train(n)

def FCNN(X_train, X_test, y_train, y_test):
    a=X_train.loc[747]
    a["2016-11-1"]=0
    a["2016-11-2"]=0
    print(a)
    print(np.reshape(a.to_numpy(),(-1,7)))

rawData = pd.read_csv('preprocessed.csv')
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

print(X.head())

#NN(X_train, X_test, y_train, y_test)    #91.307
#LR(X_train, X_test, y_train, y_test)    #91.257
#DT(X_train, X_test, y_train, y_test)    #86.264
#RF(X_train, X_test, y_train, y_test)     #91.977
#SVM(X_train, X_test, y_train, y_test)
#RBM(X_train, X_test, y_train, y_test)
#RBM(X_train, X_test, y_train, y_test)
#RBM1(X_train, X_test, y_train, y_test)
#RBMD(X_train, X_test, y_train, y_test)
FCNN(X_train, X_test, y_train, y_test)

