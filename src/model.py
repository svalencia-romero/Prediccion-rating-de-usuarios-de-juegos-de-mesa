import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
# import os
# print(os.getcwd())

df = pd.read_csv("../data/processed/bgg_proc_ml.csv")

X = df[["BGG Rank","Complexity Average","Mech_role_camp","Strategy","Wargames"]]
y = df["Rating Average"]

# Creación de train y test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.20,random_state=5)

X_train.to_csv("../data/train/X_train.csv",index=False)
y_train.to_csv("../data/train/y_train.csv",index=False)
X_test.to_csv("../data/test/X_test.csv",index=False)
y_test.to_csv("../data/test/y_test.csv",index=False)


# Modelo lineal
poly_reg = PolynomialFeatures(degree=3)
poly_reg.fit(X_train)
X_train_poly = poly_reg.transform(X_train)
pol_reg_3 = LinearRegression()

# Entrenamiento del modelo
pol_reg_3.fit(X_train_poly, y_train)

# Subida del modelo.
pickle.dump(pol_reg_3, open('../models/trained_pol_3.pkl', 'wb'))
