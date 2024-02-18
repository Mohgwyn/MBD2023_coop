from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# APARTADO 1 
# Se describe las dimensiones del dataset, contando con 30 características y
# 2 categorías, personas que generan la enfermedad y aquellas que no.
# =============================================================================

X, y = load_breast_cancer(return_X_y=True)

num_samples, num_features = X.shape
_, num_categories = np.unique(y, return_counts=True)

print('Number of samples in the dataset: ', num_samples)
print('Number of features in the dataset: ', num_features)
print('Number of categories in the dataset: ', num_categories.size)
print('Samples in first category: ', num_categories[0])
print('Samples in second category: ', num_categories[1])

# =============================================================================
# APARTADO 1 

# Con el fin de entender la naturaleza de las features se ha decidido represen-
# tar cada una de ellas junto con la label de si se padece la enfermedad o no, 
# así se pueden seleccionar aquellas características que más discriminen entre 
# las dos posibles categorías, es decir, las más representativas
# =============================================================================
objects, features = np.shape(X)
categories, frequency = np.unique(y, return_counts=True)

print(f'El dataset tiene {features} características por cada elemento.\n')
print('Hay dos categorías.\n')

for i, category in enumerate(categories):
    print(f'Categoría {category}: {frequency[i]} elementos')

# for i in range(features):
#     plt.figure()
#     plt.title(f'Feature {i}')
#     plt.scatter(X[:,i], range(objects), s=1, c=y);

# Visionando las features de manera gráfica se consideran que las más represen-
# tativas son la: 0,2,3,5,6,7,10,12,13,16,20,22,23,26,27

# Por lo tanto se va a crear un dataframe que solo contenga esas features:

df_mostWantedFeatures = pd.DataFrame(X)
columnas_no_deseadas = [1,4,8,9,11,14,15,17,18,19,21,24,25,28,29]
# Se eliminan las features no deseadas
df_mostWantedFeatures = df_mostWantedFeatures.drop(columnas_no_deseadas, axis=1)

# =============================================================================
# APARTADO 2

# En esta apartado se realiza un estudio estadistico de las features a través
# del uso de un box plot con el fin de localizar los outliers, esta información
# puede ser útil con el fin de conseguir clasificar algunas muestras dentro de
# una categoría en concreto
# =============================================================================

# Se normaliza para una mejor visualización de las features y outliers
XboxPlot = df_mostWantedFeatures # con todas las features y o con las 15 representativas? supongo que con las 15
miEscalador = MinMaxScaler()
miEscalador.fit(df_mostWantedFeatures)
XboxPlot_esc = miEscalador.transform(XboxPlot)

#plt.boxplot(X[:,23])
plt.figure()
plt.boxplot(XboxPlot_esc)

# Agregar etiquetas y título
plt.xlabel('Features') # en el gráfico no sale el numero de feature correcto, solo la enumeracion del 1 al 15
plt.ylabel('Values')
plt.title('FEATURES BOXPLOT ESCALADO')

# Mostrar el boxplot
plt.show()  ## AL RUNEAR EL CODIGO ENTERO SE RAYA, SI LO RANEAS POR SEPARADO SALE BIEN

## Handcrafted method para localizar los outlier de cada característica

def valStadistics(data,columna):
    df = pd.DataFrame(data)
    #Primer cuartil
    Q1 = df[columna].quantile(0.25)
    #Tercer cuartil
    Q3 = df[columna].quantile(0.75)
    #Rango intercuartil
    IQR = Q3-Q1
    #Bigote inferior
    BI = Q1 - 1.5 * IQR
    #Bigote superior
    BS = Q3 + 1.5 * IQR
    return BI, BS

def detectOutliers(data,columna):
    BI,BS = valStadistics(data, columna)
    df = pd.DataFrame(data)
    filasOutliers = (df[columna] < BI) | (df[columna] > BS)
    #outliers = data[filasOutliers,columna]
    outliers = df.loc[filasOutliers, [columna]]
    return outliers

# Se ha pensado que obteniendo los outliers de cada caracteristica, de aquellas,
# consideradas representativas, se podría determinar si una muestra/persona 
# padece la enfermedad si cumple la condición de ser outlier en todas las
# features.

outliers = []
for feature in [0,2,3,5,6,7,10,12,13,16,20,22,23,26,27]:
    outliers.append(detectOutliers(df_mostWantedFeatures,feature))
    
indices_comunes = set(outliers[0].index)

for df in outliers[1:]:
    indices_comunes = indices_comunes.intersection(set(df.index))    

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

print(accuracy_score(ytest,ypred))


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

print(f'miGSCV: {miGSCV.best_estimator_.get_params()}')
print(f'Accuracy con grid search kfolds {accuracy_score(ytest,ypred)}')

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


























