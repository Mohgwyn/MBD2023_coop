# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:02:07 2024

@author: nadie
"""

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

X, y = load_breast_cancer(return_X_y=True)

# =============================================================================
# Apartado 1
# 
# - Quitar features que no sean representativas?
# =============================================================================

objects, features = np.shape(X)
categories, frequency = np.unique(y, return_counts=True)

print(f'El dataset tiene {features} características por cada elemento.')
print('Hay dos categorías.\n')

for i, category in enumerate(categories):
    print(f'Categoría {category}: {frequency[i]} elementos')
    print(f'Porcentaje de clase {i}: {frequency[i]/objects*100}\n')

# for i in range(features):
#     plt.figure()
#     plt.title(f'Feature {i}')
#     plt.scatter(X[:,i], range(objects), s=1, c=y)

# =============================================================================
# Apartado 2
#
# - Mirar outlayers que se repiten entre features
# - Escalar para el boxplot?
# =============================================================================

plt.figure()
plt.boxplot(X)

def valStadistics(data,columna):
    df = pd.DataFrame(data)
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3-Q1

    BI = Q1 - 1.5 * IQR
    BS = Q3 + 1.5 * IQR
    
    return BI, BS
    
def detectOutliers(data,columna):
    BI,BS = valStadistics(data, columna)
    df = pd.DataFrame(data)
    filasOutliers = (df[columna] < BI) | (df[columna] > BS)
    outliers = df.loc[filasOutliers, [columna]]
    return outliers
    
outliers = detectOutliers(X, 0)

# =============================================================================
# Apartado 3
# 
# - Hay overfitting?
# - Quitar las features no representativas?
# =============================================================================

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X,y):
    Xtrain, ytrain = X[train_index], y[train_index]
    Xtest, ytest = X[test_index], y[test_index]
    
miEscalador = MinMaxScaler()
miEscalador.fit(Xtrain)
Xtrain_esc = miEscalador.transform(Xtrain)
Xtest_esc = miEscalador.transform(Xtest)


miK = 2
miKNN = KNeighborsClassifier(n_neighbors=miK)
miKNN.fit(Xtrain_esc,ytrain)
ypred = miKNN.predict(Xtest_esc)

print(f'accuracy para k=2: {accuracy_score(ytest,ypred)}\n')


miParamGrid = {'weights':['uniform','distance'],
               'metric':['minkowski'],
               'p':[1,2,3,4,5,6,7],
               'n_neighbors':[1,3,5,7,9,11]}

from sklearn.model_selection import GridSearchCV

miGSCV = GridSearchCV(KNeighborsClassifier(),miParamGrid,scoring='accuracy',cv=10,verbose=1,n_jobs=-1)
miGSCV.fit(Xtrain_esc,ytrain)
miMejorModelo = miGSCV.best_estimator_
miMejorModelo.fit(Xtrain_esc,ytrain)

ypred = miMejorModelo.predict(Xtest_esc)

print(f'Accuracy con grid search kfolds: {accuracy_score(ytest,ypred)}\n')

# =============================================================================
# Apartado 4
# =============================================================================

from sklearn.model_selection import LeaveOneOut

looGSCV = GridSearchCV(KNeighborsClassifier(),miParamGrid,scoring='accuracy',cv=LeaveOneOut(),verbose=1,n_jobs=-1)
looGSCV.fit(Xtrain_esc,ytrain)
miMejorModelo = miGSCV.best_estimator_
miMejorModelo.fit(Xtrain_esc,ytrain)

ypred = miMejorModelo.predict(Xtest_esc)

print(f'Accuracy con grid search loo: {accuracy_score(ytest,ypred)}\n')

# =============================================================================
# Apartado 5
# =============================================================================

# Estratificar el proceso de cross-validation es importante para evitar 
# problemas de overfitting debidos al desbalanceo del dataset. Sin embargo, 
# en nuestro dataset no hay un gran desbalanceo. La distribución de clases 
# en el dataset es la siguiente: 
    

from sklearn.model_selection import train_test_split

X_trainns, X_testns, y_trainns, y_testns = train_test_split(X[200:400, :], y[200:400], test_size=0.2, random_state=0)

objects, features = np.shape(X[200:400,:])
categories, frequency = np.unique(y[200:400], return_counts=True)

print(f'Porcentaje de clase 0: {frequency[0]/objects*100}')
print(f'Porcentaje de clase 1: {frequency[1]/objects*100}\n')

miEscalador = MinMaxScaler()
miEscalador.fit(X_trainns)
X_trainns = miEscalador.transform(X_trainns)
X_testns = miEscalador.transform(X_testns)

# miK = 2
# miKNN = KNeighborsClassifier(n_neighbors=miK)
# ypred = miMejorModelo.predict(Xtest_esc)
# miKNN.fit(Xtrain_nostratified, ytrain_nostratified)
# ypred = miKNN.predict(Xtest_nostratified)

miGSCV = GridSearchCV(KNeighborsClassifier(),miParamGrid,scoring='accuracy',cv=10,verbose=1,n_jobs=-1)
miGSCV.fit(X_trainns, y_trainns)
miMejorModelo = miGSCV.best_estimator_
miMejorModelo.fit(Xtrain_esc,ytrain)

ypredtrain = miMejorModelo.predict(X_trainns)
ypred = miMejorModelo.predict(X_testns)

print(f'Accuracy con dataset reducido en train: {accuracy_score(y_trainns, ypredtrain)}')
print(f'Accuracy con dataset reducido en test: {accuracy_score(y_testns,ypred)}\n')


























