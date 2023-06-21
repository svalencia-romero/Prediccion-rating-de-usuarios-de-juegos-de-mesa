from ast import IfExp
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
import yaml

df = pd.read_csv("../data/processed/bgg_proc_ml.csv")

X = df[["BGG Rank","Complexity Average","Mech_role_camp","Strategy","Wargames"]]
y = df["Rating Average"]

# Modelo lineal polinomio de grado 3

# Caracteristicas de modelo
with open('../models/model_config.yaml', 'r') as file:
    model_config = yaml.safe_load(file)



# Creación de train y test

poly_feats = PolynomialFeatures(degree = model_config['degree'])
poly_feats.fit(X)
X_poly = poly_feats.transform(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_poly,y,test_size= model_config['test_size'],random_state=model_config['random_state'])

X_train.to_csv("../data/train/X_train.csv",index=False)
y_train.to_csv("../data/train/y_train.csv",index=False)
X_test.to_csv("../data/test/X_test.csv",index=False)
y_test.to_csv("../data/test/y_test.csv",index=False)


# Modelo lineal

lin_reg = LinearRegression() 


# Entrenamiento del modelo

lin_reg.fit(X_train, y_train)

# Subida del modelo.
pickle.dump(lin_reg, open('../models/trained_pol_3.pkl', 'wb'))