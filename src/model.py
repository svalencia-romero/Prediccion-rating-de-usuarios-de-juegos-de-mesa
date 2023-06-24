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

# Función de entrenamiento de modelos 

def lin_reg_pol():
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

    lin_reg.fit(X_train_poly, y_train)

    # Subida del modelo.
    pickle.dump(lin_reg, open('../models/modelo_lineal/trained_pol_3.pkl', 'wb'))


def tree_dec_gs():
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


#----------------------------------------------------------------
# -----------------------   Carga df   --------------------------
# ---------------------------------------------------------------

dtype_dict = {'Mech Not Defined': "uint8", 'Mech_Acting': "uint8", 'Mech_Action':"uint8", 'Mech_tokens':"uint8",
       'Mech_construcc_farm':"uint8", 'Mech_roll_thng':"uint8", 'Mech_cards':"uint8", 'Mech_role_camp':"uint8",
       'Mech_board':"uint8", 'Mech_money':"uint8", 'Mech_score':"uint8", 'Mech_turnbased':"uint8", 'Mech_team':"uint8",
       'Mech_skill':"uint8", 'Mech_solo':"uint8", 'Abstract':"uint8", 'Children':"uint8", 'Customizable':"uint8",
       'Family':"uint8", 'Party':"uint8", 'Strategy':"uint8", 'Thematic':"uint8", 'Wargames':"uint8",
       'Domain_Not Defined':"uint8"
        }
df = pd.read_csv("../data/processed/bgg_proc_ml.csv", dtype = dtype_dict)

X = df[["BGG Rank","Complexity Average","Mech_role_camp","Strategy","Wargames"]]
y = df["Rating Average"]


#----------------------------------------------------------------
# -----------------------   Selector   --------------------------
# ---------------------------------------------------------------

selector = input("¿Quieres un entrenar un modelo en particular(M) o quieres entrenar todos(Cualquier tecla)?:(M/Cualquier tecla)")

if selector == "M":
    selector_2 = input("¿Que módelo quieres entrenar? Lineal(L) -- Arbol de decision(A)")
    if selector_2 == "L":
        print("Entrenando modelo lineal...")
        lin_reg_pol()
        print("Entrenamiento modelo lineal completado")
        time.sleep(5)
        
    if selector_2 == "A":
        print("Entrenando modelo arbol de decision...")
        tree_dec_gs()
        print("Entrenamiento modelo arbol de decisión completado")
        time.sleep(5)

else:
    print("Entrenando modelo lineal...")
    lin_reg_pol()
    print("Entrenamiento modelo lineal completado")
    print("Entrenando modelo arbol de decisión...")
    tree_dec_gs()
    print("Entrenamiento modelo arbol de decisión completado")
    time.sleep(5)


