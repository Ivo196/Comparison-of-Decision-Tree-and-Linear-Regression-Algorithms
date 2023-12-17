# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:20:17 2023

@author: ivoto
"""

import numpy as np
import pandas as pd
sensorData = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', usecols=range(15))
sensorData = sensorData.dropna(how='all', axis=0)
sensorData.replace(-200, np.nan, inplace=True)
sensorData.shape #Optenemos la cantidad de instancias(Datos Guardados)
sensorData.columns #Nombre de las columnas
sensorData.info() #Genera informacion general, podemos ver si hay valores no nulos
sensorData.describe()

#X = sensorData[["PT08.S1(CO)"]]
X = sensorData.drop(["C6H6(GT)","Date","Time"],axis=1)
X.fillna(X.mean(numeric_only=True), inplace=True)
y = sensorData[["C6H6(GT)"]]
y.fillna(y.mean(numeric_only=True), inplace=True)

#Dividimos en X_train, X_Test y_train y_test 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )



# Redimensionar X_train y X_test a 2D
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)


#Crear modelo de Regresion Lineal Simple 

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)


##Arbol de decision
from sklearn.tree import DecisionTreeRegressor

# Crear el modelo de árbol de decisión para regresión
modelo_arbol_decision = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features=None,  # Usar todas las características
    criterion='poisson',
)

# Entrenar el modelo
modelo_arbol_decision.fit(X_train, y_train)

# Evaluar el modelo
puntaje = modelo_arbol_decision.score(X_test, y_test)
# Generar predicciones para el conjunto completo de datos de prueba
y_pred = modelo_arbol_decision.predict(X_test)
y_pred = pd.DataFrame(y_pred)



































