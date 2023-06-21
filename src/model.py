import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
import yaml
import time

df = pd.read_csv("../data/processed/bgg_proc_ml.csv")

X = df[["BGG Rank","Complexity Average","Mech_role_camp","Strategy","Wargames"]]
y = df["Rating Average"]

# Modelo lineal polinomio de grado 3

# Caracteristicas de modelo

with open('../models/model_config.yaml', 'r') as file:
    model_config = yaml.safe_load(file)


# Creación de train y test

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= model_config['test_size'],random_state=model_config['random_state'])

poly_feats = PolynomialFeatures(degree = model_config['degree'])
poly_feats.fit(X_train)
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
pickle.dump(lin_reg, open('../models/trained_pol_3.pkl', 'wb'))

print("Entrenamiento completado")
time.sleep(5)
