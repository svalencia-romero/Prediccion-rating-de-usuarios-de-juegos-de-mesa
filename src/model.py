import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import pickle
import time
import functions

df = pd.read_csv("../data/processed/bgg_proc_ml.csv")

X = df[["BGG Rank","Complexity Average","Mech_role_camp","Strategy","Wargames"]]
y = df["Rating Average"]

# ------------------------------------ Modelo lineal polinomio de grado 3 ---------------------------------------

# Caracteristicas de modelo

model_config_path_lin = '../models/modelo_lineal/model_config_lin.yaml'

lin_model_conf = functions.load_config(model_config_path_lin)


# Creación de train y test

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= lin_model_conf['test_size'],random_state=lin_model_conf['random_state'])

poly_feats = PolynomialFeatures(degree = lin_model_conf['degree'])
poly_feats.fit(X_train)

# Salvamos el modelo polinomico para después utilizarlo
pickle.dump(poly_feats, open('../models/modelo_lineal/transformacion_polinomio.pkl', 'wb'))
  

X_train_poly = poly_feats.transform(X_train)
X_test_poly = poly_feats.transform(X_test) 

#Transformador Data Frame de Train y test

df_train = pd.DataFrame(X_train)
df_train['Rating Average'] = y_train

df_test = pd.DataFrame(X_test)
df_test['Rating Average'] = y_test

df_train.to_csv('../data/train/train.csv', index=False)
df_test.to_csv('../data/test/test.csv', index=False)


# Modelo lineal

lin_reg = LinearRegression() 

# Entrenamiento del modelo

print("Entrenando modelo...")
lin_reg.fit(X_train_poly, y_train)

# Subida del modelo.
pickle.dump(lin_reg, open('../models/modelo_lineal/trained_pol_3.pkl', 'wb'))


# Y si hago un if con selección de modelos a entrenar? todos, uno solo...
# print("Entrenamiento completado")
# time.sleep(5)  


# --------------------------------------------  Modelo decision tree ----------------------------------------------

model_config_path_tree = "../models/arbol_decision/model_config_dtr.yaml"

dtr_gs_model_conf = functions.load_config(model_config_path_tree)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= dtr_gs_model_conf['test_size'],random_state=dtr_gs_model_conf['random_state'])

# Crear el estimador DecisionTreeRegressor
estimator = DecisionTreeRegressor(random_state=5)

# Crear el objeto GridSearchCV con la configuración cargada
dtr_gs = GridSearchCV(estimator, dtr_gs_model_conf['GridSearchCV']['param_grid'], cv=dtr_gs_model_conf['GridSearchCV']['cv'],
                           scoring=dtr_gs_model_conf['GridSearchCV']['scoring'])

# Realizar la búsqueda de parámetros

dtr_gs.fit(X_train, y_train)

# Subida del modelo.
pickle.dump(dtr_gs, open('../models/arbol_decision/dtr_gs.pkl', 'wb'))

print("Entrenamiento completado")
time.sleep(5)  



