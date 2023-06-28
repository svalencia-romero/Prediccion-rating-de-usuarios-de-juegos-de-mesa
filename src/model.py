import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
import pickle
import time
import functions
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings

warnings.simplefilter("ignore")

# Función de entrenamiento de modelos 

def lin_reg_pol():
    #------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Modelo lineal polinomio de grado 3 -----------------------------------------
    #------------------------------------------------------------------------------------------------------------------

    # Caracteristicas de modelo

    model_config_path_lin = '../models/modelo_lineal/model_config_lin.yaml'

    lin_model_conf = functions.load_config(model_config_path_lin)

    # Creación de train y test

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

    # Crear el estimador DecisionTreeRegressor
    model = AdaBoostRegressor(random_state=5)

    # Crear el objeto GridSearchCV con la configuración cargada
    rnd_ft = GridSearchCV(model, ada_gs_model_conf['GridSearchCV']['param_grid'], cv=ada_gs_model_conf['GridSearchCV']['cv'],
                            scoring=ada_gs_model_conf['GridSearchCV']['scoring'])

    # Realizar la búsqueda de parámetros
    rnd_ft.fit(X_train, y_train)

    # Subida del modelo.
    pickle.dump(rnd_ft, open('../models/ada_gs/ada_gs.pkl', 'wb'))

def gbrt():
    #------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------  Modelo Gradient Boosting Regressor---------------------------------
    #------------------------------------------------------------------------------------------------------------------
    model_config_path_tree = "../models/gbrt/model_config_gbrt.yaml"

    gbrt_model_conf = functions.load_config(model_config_path_tree)

    # Crear el estimador DecisionTreeRegressor
    model = GradientBoostingRegressor(random_state=5)

    # Crear el objeto GridSearchCV con la configuración cargada
    gbrt = GridSearchCV(model, gbrt_model_conf['GridSearchCV']['param_grid'], cv=gbrt_model_conf['GridSearchCV']['cv'],
                            scoring=gbrt_model_conf['GridSearchCV']['scoring'])

    # Realizar la búsqueda de parámetros
    gbrt.fit(X_train, y_train)

    # Subida del modelo.
    pickle.dump(gbrt, open('../models/gbrt/gbrt.pkl', 'wb'))

def pca_rf():
    #------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------  Modelo PCA con Random Forest --------------------------------------
    #------------------------------------------------------------------------------------------------------------------
    model_config_path_tree = "../models/pca_rf/model_config_pca_rf.yaml"

    pca_rf_model_conf = functions.load_config(model_config_path_tree)

    # Configurar el pipeline
    steps = []
    for step_name, step_class, step_params in pca_rf_model_conf['steps']:
        step_instance = eval(step_class)(**step_params) if step_params else eval(step_class)()
        steps.append((step_name, step_instance))
    pipe_gs = Pipeline(steps)

    # Configurar la búsqueda de hiperparámetros
    params = pca_rf_model_conf['params']
    gs = GridSearchCV(pipe_gs, params, cv=pca_rf_model_conf['cv'], scoring=pca_rf_model_conf['scoring'])

    # Entrenar y ajustar el modelo
    gs.fit(X_train, y_train)

    # Subida del modelo.
    pickle.dump(gs, open('../models/pca_rf/pca_rf.pkl', 'wb'))

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

X = df.drop(['ID','Name','Year Published','Rating Average','Users Rated','Mechanics','Domains'],axis=1)
y = df['Rating Average']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size= 0.20 ,random_state=5)

df_train = pd.DataFrame(X_train, columns=['Min Players', 'Max Players', 'Play Time', 'Min Age', 'BGG Rank',
       'Complexity Average', 'Owned Users', 'Mech Not Defined', 'Mech_Acting',
       'Mech_Action', 'Mech_tokens', 'Mech_construcc_farm', 'Mech_roll_thng',
       'Mech_cards', 'Mech_role_camp', 'Mech_board', 'Mech_money',
       'Mech_score', 'Mech_turnbased', 'Mech_team', 'Mech_skill', 'Mech_solo',
       'Abstract', 'Children', 'Customizable', 'Family', 'Party', 'Strategy',
       'Thematic', 'Wargames', 'Domain_Not Defined'])
df_train['Rating Average'] = y_train

df_test = pd.DataFrame(X_test, columns=['Min Players', 'Max Players', 'Play Time', 'Min Age', 'BGG Rank',
       'Complexity Average', 'Owned Users', 'Mech Not Defined', 'Mech_Acting',
       'Mech_Action', 'Mech_tokens', 'Mech_construcc_farm', 'Mech_roll_thng',
       'Mech_cards', 'Mech_role_camp', 'Mech_board', 'Mech_money',
       'Mech_score', 'Mech_turnbased', 'Mech_team', 'Mech_skill', 'Mech_solo',
       'Abstract', 'Children', 'Customizable', 'Family', 'Party', 'Strategy',
       'Thematic', 'Wargames', 'Domain_Not Defined'])
df_test['Rating Average'] = y_test

df_train.to_csv('../data/train/train.csv', index=False)
df_test.to_csv('../data/test/test.csv', index=False)

#----------------------------------------------------------------
# -----------------------   Selector   --------------------------
# ---------------------------------------------------------------

selector = input("¿Quieres un entrenar un modelo en particular(M) o quieres entrenar todos(Cualquier tecla)?:(M/Cualquier tecla)")

if selector == "M":
    selector_2 = input("¿Que módelo quieres entrenar? \n Lineal(L) (10seg) \n Arbol de decision(D) (10 min aprox) \n Random Forest(R) (3 min aprox) \n Ada Boost(A) (3 min aprox) \n Gradient Boosting Regressor(G) (10 min aprox) \n PCA con Random Forest Regressor(P) \n (L\D\R\A\G\P): ")
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
    
    if selector_2 == "G":
        print("Entrenando modelo Gradient Boosting Regressor...")
        print("10 minutos aproximadamente de entrenamiento...paciencia...")
        gbrt()
        print("Entrenamiento modelo Gradient Boosting Regressor completado")
        time.sleep(5)

    if selector_2 == "P":
        print("Entrenando modelo PCA con Random Forest Regressor ...")
        print("15 minutos aproximadamente de entrenamiento...paciencia...")
        pca_rf()
        print("Entrenamiento modelo PCA con Random Forest Regressor completado")
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
    # Entrenamiento modelo Gradient Boosting Regressor
    print("Entrenando modelo Gradient Boosting Regressor...")
    print("10 minutos aproximadamente de entrenamiento...paciencia...")
    gbrt()
    print("Entrenamiento modelo Gradient Boosting Regressor completado")
    # Entrenamiento modelo PCA con Random Forest Regressor
    print("Entrenando modelo PCA con Random Forest Regressor ...")
    print("15 minutos aproximadamente de entrenamiento...paciencia...")
    pca_rf()
    print("Entrenamiento modelo PCA con Random Forest Regressor completado")
    time.sleep(5)


