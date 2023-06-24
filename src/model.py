import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
import pickle
import time
import functions

# Función de entrenamiento de modelos 

def lin_reg_pol():
    #------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Modelo lineal polinomio de grado 3 -----------------------------------------
    #------------------------------------------------------------------------------------------------------------------

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
    #------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------  Modelo decision tree ----------------------------------------------
    #------------------------------------------------------------------------------------------------------------------

    model_config_path_tree = "../models/arbol_decision/model_config_dtr.yaml"

    dtr_gs_model_conf = functions.load_config(model_config_path_tree)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= dtr_gs_model_conf['test_size'],random_state=dtr_gs_model_conf['random_state'])

    # Crear el estimador DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=5)

    # Crear el objeto GridSearchCV con la configuración cargada
    dtr_gs = GridSearchCV(model, dtr_gs_model_conf['GridSearchCV']['param_grid'], cv=dtr_gs_model_conf['GridSearchCV']['cv'],
                            scoring=dtr_gs_model_conf['GridSearchCV']['scoring'])

    # Realizar la búsqueda de parámetros
    dtr_gs.fit(X_train, y_train)

    # Subida del modelo.
    pickle.dump(dtr_gs, open('../models/arbol_decision/dtr_gs.pkl', 'wb'))

def rnd_ft():
    #------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------  Modelo Random Forest ----------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    model_config_path_tree = "../models/random_forest/model_config_rnd_ft.yaml"

    rnd_ft_model_conf = functions.load_config(model_config_path_tree)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= rnd_ft_model_conf['test_size'],random_state=rnd_ft_model_conf['random_state'])

    # Crear el estimador DecisionTreeRegressor
    model = RandomForestRegressor(random_state=5)

    # Crear el objeto GridSearchCV con la configuración cargada
    rnd_ft = GridSearchCV(model, rnd_ft_model_conf['GridSearchCV']['param_grid'], cv=rnd_ft_model_conf['GridSearchCV']['cv'],
                            scoring=rnd_ft_model_conf['GridSearchCV']['scoring'])

    # Realizar la búsqueda de parámetros
    rnd_ft.fit(X_train, y_train)

    # Subida del modelo.
    pickle.dump(rnd_ft, open('../models/random_forest/rnd_ft.pkl', 'wb'))
def ada_gs():
    #------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------  Modelo Ada Boost ----------------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    model_config_path_tree = "../models/ada_gs/model_config_ada_gs.yaml"

    ada_gs_model_conf = functions.load_config(model_config_path_tree)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= ada_gs_model_conf['test_size'],random_state=ada_gs_model_conf['random_state'])

    # Crear el estimador DecisionTreeRegressor
    model = AdaBoostRegressor(random_state=5)

    # Crear el objeto GridSearchCV con la configuración cargada
    rnd_ft = GridSearchCV(model, ada_gs_model_conf['GridSearchCV']['param_grid'], cv=ada_gs_model_conf['GridSearchCV']['cv'],
                            scoring=ada_gs_model_conf['GridSearchCV']['scoring'])

    # Realizar la búsqueda de parámetros
    rnd_ft.fit(X_train, y_train)

    # Subida del modelo.
    pickle.dump(rnd_ft, open('../models/ada_gs/ada_gs.pkl', 'wb'))

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

X = df.drop(["Name","Rating Average","Domains", "Mechanics"],axis=1)
y = df["Rating Average"]


#----------------------------------------------------------------
# -----------------------   Selector   --------------------------
# ---------------------------------------------------------------

selector = input("¿Quieres un entrenar un modelo en particular(M) o quieres entrenar todos(Cualquier tecla)?:(M/Cualquier tecla)")

if selector == "M":
    selector_2 = input("¿Que módelo quieres entrenar? Lineal(L) (10seg) -- Arbol de decision(D) (10 min aprox) -- Random Forest(R) (3 min aprox) -- Ada Boost(A) (3 min aprox)")
    if selector_2 == "L":
        print("Entrenando modelo lineal...")
        lin_reg_pol()
        print("Entrenamiento modelo lineal completado")
        time.sleep(5)
        
    if selector_2 == "D":
        print("Entrenando modelo arbol de decision...")
        print("10 minutos aproximadamente de entrenamiento...paciencia...")
        tree_dec_gs()
        print("Entrenamiento modelo arbol de decisión completado")
        time.sleep(5)
    
    if selector_2 == "R":
        print("Entrenando modelo random forest...")
        print("3 minutos aproximadamente de entrenamiento...paciencia...")
        rnd_ft()
        print("Entrenamiento modelo arbol de decisión completado")
        time.sleep(5)
    
    if selector_2 == "A":
        print("Entrenando modelo Ada Boost...")
        print("3 minutos aproximadamente de entrenamiento...paciencia...")
        ada_gs()
        print("Entrenamiento modelo arbol de decisión completado")
        time.sleep(5)
    
    

else:
    # Entrenamiento modelo lineal
    print("Entrenando modelo lineal...")
    lin_reg_pol()
    print("Entrenamiento modelo lineal completado")
    # Entrenamiento modelo arbol
    print("Entrenando modelo arbol de decision...")
    print("10 minutos aproximadamente de entrenamiento...paciencia...")
    tree_dec_gs()
    print("Entrenamiento modelo arbol de decisión completado")
    # Entrenamiento modelo Random forest
    print("Entrenando modelo Random forest...")
    print("3 minutos aproximadamente de entrenamiento...paciencia...")
    rnd_ft()
    print("Entrenamiento modelo arbol de decisión completado")
    # Entrenamiento modelo Ada Boost
    print("Entrenando modelo Ada Boost...")
    print("3 minutos aproximadamente de entrenamiento...paciencia...")
    ada_gs()
    print("Entrenamiento modelo arbol de decisión completado")
    time.sleep(5)


