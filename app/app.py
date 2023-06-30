import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 
from sklearn.preprocessing import PolynomialFeatures
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

st.set_option('deprecation.showPyplotGlobalUse', False)

# Carga de test y otros archivos

with open(os.path.join(dir_path,"..","docs","Mechanics_selec.txt"), "r") as file:
        txt_mecanicas = file.read()
df_test = pd.read_csv(os.path.join(dir_path, "..", "data", "test", "test.csv"))
df_ml_original = pd.read_csv(os.path.join(dir_path, "..", "data", "processed", "bgg_proc_ml.csv"))
df_raw = pd.read_csv(os.path.join(dir_path, "..", "data", "raw", "bgg_dataset.csv"),sep=";")

# Obtener las características (X_test) y las etiquetas (y_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']

df_train = pd.read_csv(os.path.join(dir_path, "..", "data", "train", "train.csv"))
X_train = df_train.drop('Rating Average', axis=1)

# Carga de configuraciones

with open(os.path.join(dir_path,"..","models","bagging_regressor","model_config_bag_gs.yaml"), "r") as file:
    model_bag = yaml.safe_load(file)

with open(os.path.join(dir_path,"..","models","modelo_lineal","model_config_lin.yaml"),"r") as file:
    model_lin = yaml.safe_load(file)

with open(os.path.join(dir_path,"..","models","arbol_decision","model_config_dtr.yaml"), "r") as file:
    model_cfg_dtr = yaml.safe_load(file)

with open(os.path.join(dir_path,"..","models","gbrt","model_config_gbrt.yaml"), "r") as file:
    model_cfg_gbrt = yaml.safe_load(file)

with open(os.path.join(dir_path,"..","models","pca_rf","model_config_pca_rf.yaml"), "r")as file:
    model_cfg_pca_rf = yaml.safe_load(file)

with open(os.path.join(dir_path,"..","models","random_forest","model_config_rnd_ft.yaml"), "r") as file:
    model_cfg_rnd_ft = yaml.safe_load(file)

with open(os.path.join(dir_path,"..","models","ada_gs","model_config_ada_gs.yaml"), "r") as file:
    model_cfg_ada_gs = yaml.safe_load(file)

# Carga de modelos
with open(os.path.join(dir_path, "..", "models", "bagging_regressor", "bar_reg.pkl"),"rb") as li:
    bag_reg = pickle.load(li)

with open(os.path.join(dir_path, "..", "models", "modelo_lineal", "trained_pol_3.pkl"),"rb") as li:
    lin_reg = pickle.load(li)

with open(os.path.join(dir_path, "..", "models", "modelo_lineal", "transformacion_polinomio.pkl"),"rb") as f:
    transformacion = pickle.load(f)

with open(os.path.join(dir_path, "..", "models", "arbol_decision", "dtr_gs.pkl"),"rb") as dtr:
    dtr_gs = pickle.load(dtr)

with open(os.path.join(dir_path, "..", "models", "random_forest", "rnd_ft.pkl"),"rb") as dtr:
    rnd_ft = pickle.load(dtr)    

with open(os.path.join(dir_path, "..", "models", "ada_gs", "ada_gs.pkl"),"rb") as ada:
    ada_gs = pickle.load(ada)

with open(os.path.join(dir_path, "..", "models", "gbrt", "gbrt.pkl"),"rb") as gbrt:
    gbrt = pickle.load(gbrt)

with open(os.path.join(dir_path, "..", "models", "pca_rf", "pca_rf.pkl"),"rb") as pca:
    pca_rf = pickle.load(pca)

df_errores = pd.read_csv(os.path.join(dir_path, "..", "data", "processed", "analisis_metricas.csv"),index_col="Métricas")



#Función graficas errores 

# Crear la gráfica
def plot_barplots(df, x, variables):
    num_plots = len(variables)
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 4))
    
    for i, var in enumerate(variables):
        sns.barplot(data=df, x=x, y=var, label=var, orient='h', ax=axes[i])
        axes[i].set_xlabel(x)
        axes[i].set_ylabel(var)
        axes[i].set_xlim(6, 8)
    
    plt.tight_layout()
    st.pyplot(fig)

def var_corr(df,variables,titulo,ejex):
    sns.set(style="darkgrid")
    fig = None
    for var in variables:
        if fig is None:
            fig = sns.kdeplot(df[var], fill=True, label=var)
        else:
            sns.kdeplot(df[var], fill=True, label=var)

    plt.xlabel(ejex)
    plt.ylabel('Concentración de Variables')
    plt.legend()

    st.pyplot(fig.figure)

def graf_univ(var,titulo):
    fig = plt.figure(figsize=(5, 5))
    sns.set(style="darkgrid")
    sns.histplot(data=df_ml_original, x=var, color="skyblue", kde=True, bins=30)
    plt.title(titulo)

    st.pyplot(fig) 

def graf_bivar(var1,var2,titulo):
    fig = plt.figure(figsize=(8, 5))
    sns.set(style="darkgrid")
    sns.regplot(x=var1, y=var2)
    plt.title(titulo)
    st.pyplot(fig)


def grafica(predicciones,titulo):
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(y_test, predicciones, color='blue', label='Valores de prueba vs. Valores predichos')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Línea de referencia: valores reales = valores predichos
    ax.scatter(y_test, y_test, color='red', label='Valores de prueba')
    ax.set_xlabel('Valores de prueba')
    ax.set_ylabel('Valores predichos')
    ax.set_title(titulo)
    ax.legend()
    st.pyplot(fig)

def mtx_corr():
    correlation_matrix = df_ml_original.corr()

    st.write("Matriz de correlación:")
    st.write(correlation_matrix)

def grf_corr():

    plt.figure(figsize=(30, 15))
    sns.heatmap(correlation_matrix, annot=True, cmap="viridis")
    plt.title("Matriz de correlación")
    st.pyplot()


# Funcion principal del dataframe que vamos modificando con slides

def user_input_parameters(): # Parametros que metemos en la app
    min_players = st.sidebar.slider("Mínimo número de jugadores",1,10)
    max_players = st.sidebar.slider("Máximo número de jugadores",1,20)
    play_time = st.sidebar.slider("Tiempo de juego aproximado",5,150)
    min_age = st.sidebar.slider("Edad mínima",0,25)
    bgg_rank = st.sidebar.slider("Rango aproximado en BGG que crees que puede tener tu juego",1,20344,10000)
    complejidad_juego = st.sidebar.slider("Grado de complejidad",0,5)
    owned_users = st.sidebar.slider("Usuarios que estimas que tendrá el juego",1,150000,10000)
    mech_not_defined = st.sidebar.slider("Mecanicas no definidas",0,1)
    mech_acting = st.sidebar.slider("Mecanicas de acting",0,1)
    mech_action = st.sidebar.slider("Mecanicas de acción",0,1)
    mech_tokens = st.sidebar.slider("Mecanicas de tokens",0,1)
    mech_construcc_farm = st.sidebar.slider("Mecanicas de construccion y farmeo",0,1)
    mech_roll_thng = st.sidebar.slider("Mecanicas de tirar dados",0,1)
    mech_cards = st.sidebar.slider("Mecanicas de cartas",0,1)
    mech_role_camp = st.sidebar.slider("Mecanicas de rol y campaña",0,1)
    mech_board = st.sidebar.slider("Mecanicas de tablero",0,1)
    mech_money = st.sidebar.slider("Mecanicas de dinero",0,1)
    mech_score = st.sidebar.slider("Mecanicas de puntuacion",0,1)
    mech_turnbased = st.sidebar.slider("Mecanicas de turnos",0,1)
    mech_team = st.sidebar.slider("Mecanicas de equipo",0,1)
    mech_skill = st.sidebar.slider("Mecanicas de habilidad",0,1)
    mech_solo = st.sidebar.slider("Mecanicas solitario",0,1)
    abstract = st.sidebar.slider("Juegos abstractos",0,1)
    children = st.sidebar.slider("Juegos para niños",0,1)
    customizable = st.sidebar.slider("Juegos con piezas",0,1)
    family = st.sidebar.slider("Juegos de familia",0,1)
    party = st.sidebar.slider("Juegos de party",0,1)
    strategy = st.sidebar.slider("Juegos de estrategia",0,1)
    thematic = st.sidebar.slider("Juegos de tematicas concretas",0,1)
    wargames = st.sidebar.slider("Juegos de guerra",0,1)
    domain_not_defined = st.sidebar.slider("Género no definido",0,1)      
    
    

    data ={"Min Players":min_players,
        "Max Players":max_players,
        "Play Time":play_time,
        "Min Age":min_age,
        "BGG Rank":bgg_rank,
        "Complexity Average":complejidad_juego,
        "Owned Users":owned_users,
        "Mech Not Defined":mech_not_defined,
        "Mech_Acting":mech_acting,
        "Mech_Action":mech_action,
        "Mech_tokens":mech_tokens,
        "Mech_construcc_farm":mech_construcc_farm,
        "Mech_roll_thng":mech_roll_thng,
        "Mech_cards":mech_cards,
        "Mech_role_camp":mech_role_camp,
        "Mech_board":mech_board,
        "Mech_money":mech_money,
        "Mech_score":mech_score,
        "Mech_turnbased":mech_turnbased,
        "Mech_team":mech_team,
        "Mech_skill":mech_skill,
        "Mech_solo":mech_solo,
        "Abstract":abstract,
        "Children":children,
        "Customizable":customizable,
        "Family":family,
        "Party":party,
        "Strategy":strategy,
        "Thematic":thematic,
        "Wargames":wargames,            
        "Domain_Not Defined":domain_not_defined,
        }
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_parameters()

correlation_matrix = df_ml_original.corr()

# Funcion principal de la App


def main():    
    # Titulo
    st.title("Predicción Rating Usuarios Juegos de Mesa")
    st.image(os.path.join(dir_path,"..","img","board.jpg"), width=650)
    st.write("¡Bienvenido!")
    # Página para científicos de datos
    def user_page():
        st.title("Escoja la opción que desee")
        st.write()
        st.write('- Primero selecciona las caracteristicas de tu juego de mesa y despues pincha en la pestaña "Predicción de rating de usuarios"')
        show_rating = st.checkbox("Predicción de rating de usuarios")
        st.write('- En este apartado podrás comprobar diferentes cuestiones económicas importantes sobre tu juego de mesa.')   
        show_money = st.checkbox("Ganancias e inversión en marketing.")
                       
        if show_rating:
            prediccion = rnd_ft.best_estimator_.predict(df)
            st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))      
        
        if show_money: # Con esto tendremos un nivel de dinero de inversión en marketing
            
            if df["BGG Rank"].values[0] > 10000:
                cost_per_rank = 1
            else:
                cost_per_rank = 10
            
            coste_de_juego = st.number_input("¿Cuanto va a costar tu juego en la tienda?",value=25,step=1) # Coste
            cost_per_user_tienda = coste_de_juego * 0.20 # Lo que se lleva la tienda por juego vendido.
            coste_manuf_game = st.number_input("¿Cuanto dinero tienes previsto invertir en fabricar tu juego?",value=5000,step=1)
            gan_per_game = (coste_de_juego * 0.80) - cost_per_rank # Lo que se lleva el cliente por juego.
            gan_game_overall = (df["Owned Users"].values[0] * gan_per_game)  - coste_manuf_game
            inv_rank = 20345 - df["BGG Rank"].values[0]
            total_cost = inv_rank * cost_per_rank + df["Owned Users"].values[0] * cost_per_user_tienda
            inversion_total = total_cost + coste_manuf_game
            st.write("Dinero obtenido por cada juego:", round(gan_per_game, 2),"€ *")
            st.write("Total del dinero ganado con el juego :", round(gan_game_overall, 2),"€ **")
            st.write("El coste aproximado total de publicidad es de :", round(total_cost, 2),"€ ***")
            st.write("El coste aproximado total de toda la inversion en el juego es de :", round(inversion_total, 2),"€ ****")
            st.write('*Estimación realizada si las tiendas reciben el 20 % de las ventas de cada juego y también sumando el precio del rango en BGG(1€p/r 10000-20344,10€p/r 1-10000)')
            st.write('** Nº de personas que estimas que tendrá el juego * (Ganancia de cada juego - 20% tienda) - Dinero invertido en el juego')
            st.write('*** Cada rango de BGG del 10000-20344 cuesta 1€, del 1-10000, 10€ por rango. A esto hay que sumar el 20% que se lleva cada tienda.')
            st.write('**** Coste total de publicidad + inversion en fabricar el juego')

    def data_scientist_page():
        st.title("Página para Científicos de Datos")
        
        option =["Regresion lineal",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Ada Boost Regressor",
                "Gradient Boosting Regressor",
                "PCA con Random Forest Regressor",
                "Bagging Regressor"]
        model = st.sidebar.selectbox("¿Que modelo quieres probar?",option)

        show_dataframes = st.checkbox("Mostrar/ocultar Data Frame original vs Data Frame análisis")
        show_dataframe_errors = st.checkbox("Mostrar/ocultar Data Frame con métricas de cada modelo")
        show_rating = st.checkbox("Mostrar/ocultar rating de usuarios")
        show_grafs = st.checkbox("Graficas valores de prueba vs valores predichos")
        show_config = st.checkbox("Mostrar/ocultar configuraciones de modelos")
        show_corr_mat = st.checkbox("Mostrar/ocultar matriz de correlación")
        show_corr_map = st.checkbox("Mostrar/ocultar mapa de correlación")
        show_graf_bi = st.checkbox("Mostrar/ocultar comparación variables")
        
        if show_corr_map: # Matriz de correlación
                grf_corr()
        if show_corr_mat: # Heatmap
                mtx_corr()
        
        if show_graf_bi: # Graficas de EDA
            select_graf_bi_options = ("BGG rank vs Rating Usuarios","Complejidad de juego vs Rating Usuarios","Densidad Rating Average","Mecanicas que más correlan con el rating","Mecanicas con mejor nota","Géneros que más correlan con el rating","Géneros con mejor nota")
            select_graf_bi = st.selectbox("Selecciona gráfico",select_graf_bi_options)
            if select_graf_bi == "BGG rank vs Rating Usuarios":
                graf_bivar(df_ml_original["BGG Rank"],df_ml_original["Rating Average"], "BBG Rank vs Rating usuarios")
            if select_graf_bi == "Complejidad de juego vs Rating Usuarios":
                graf_bivar(df_ml_original["Complexity Average"],df_ml_original["Rating Average"], "Complejidad vs Rating Usuarios")
            if select_graf_bi == "Densidad Rating Average":
                graf_univ(df_ml_original["Rating Average"], "Densidad Rating Usuarios")
            if select_graf_bi == "Mecanicas que más correlan con el rating":
                var_corr(df_ml_original,['Mech_role_camp', 'Mech_team', 'Mech_solo'],"Variables de Mecanicas","Mecanicas")
            if select_graf_bi == "Mecanicas con mejor nota":
                plot_barplots(df_ml_original,"Rating Average",['Mech_role_camp', 'Mech_team', 'Mech_solo'])  
            if select_graf_bi == "Géneros que más correlan con el rating":
                var_corr(df_ml_original,['Strategy','Wargames'],"Variables de Géneros","Géneros")
            if select_graf_bi == "Géneros con mejor nota":
                plot_barplots(df_ml_original,"Rating Average",['Strategy','Wargames'])

        
        if show_grafs: # Graficas valores pred vs valores test
            pred_rnd_ft = rnd_ft.predict(X_test)
            poly_feats = PolynomialFeatures(degree = model_lin['degree'])
            poly_feats.fit(X_train)
            X_test_poly = poly_feats.transform(X_test)
            pred_lin = lin_reg.predict(X_test_poly)
            pred_bag = bag_reg.predict(X_test)
            pred_ada = ada_gs.predict(X_test)
            pred_gbr = gbrt.predict(X_test)
            pred_pca = pca_rf.predict(X_test)
            pred_dtr = dtr_gs.predict(X_test)

            graf_options = ["Regresion lineal",
                            "Decision Tree Regressor",
                            "Random Forest Regressor",
                            "Ada Boost Regressor",
                            "Gradient Boosting Regressor",
                            "PCA con Random Forest Regressor",
                            "Bagging Regressor"]
            graf = st.selectbox("Seleccionar grafico", graf_options)
            if graf == "Bagging Regressor":
                grafica(pred_bag,"Bagging Regressor")         
            if graf == "Regresion lineal":
                grafica(pred_lin,"Regresion lineal") 
            if graf == "Decision Tree Regressor":   
                grafica(pred_dtr,"Decision Tree Regressor")
            if graf == "Ada Boost Regressor":
                grafica(pred_ada,"Ada Boost Regressor")
            if graf == "Gradient Boosting Regressor":
                grafica(pred_gbr,"Gradient Boosting Regressor")
            if graf == "PCA con Random Forest Regressor":
                grafica(pred_pca,"PCA con Random Forest Regressor")
            if graf == "Random Forest Regressor":
                grafica(pred_rnd_ft,"Random Forest Regressor")  
        
        if show_config: # Configuración de modelos
            cfg_options = ["Regresion lineal",
                            "Decision Tree Regressor",
                            "Random Forest Regressor",
                            "Ada Boost Regressor",
                            "Gradient Boosting Regressor",
                            "PCA con Random Forest Regressor",
                            "Bagging Regressor"]
            cfg = st.selectbox("Seleccionar configuración", cfg_options)        
            if cfg == "Bagging Regressor":   
                st.write(model_bag)
            if cfg == "Decision Tree Regressor":   
                st.write(model_cfg_dtr)
            if cfg == "Ada Boost Regressor":
                st.write(model_cfg_ada_gs)
            if cfg == "Gradient Boosting Regressor":
                st.write(model_cfg_gbrt)
            if cfg == "PCA con Random Forest Regressor":
                st.write(model_cfg_pca_rf)
            if cfg == "Random Forest Regressor":
                st.write(model_cfg_rnd_ft)
            if cfg == "Regresion lineal":
                st.write(model_lin)
   
        if show_dataframes: # Muestra de df original vs mod
                st.write('-- Dataset Original --')
                st.dataframe(df_raw)
                st.write('-- Dataset Modificado --')
                st.dataframe(df_ml_original)
                text_show = st.checkbox("Mostrar/ocultar Seleccion de mecánicas")
                if text_show:
                    st.text_area("Proceso de selección de mecánicas", txt_mecanicas)

        if show_dataframe_errors: # Metricas
                st.dataframe(df_errores)  

        if show_rating: # Mostrar rating de usuarios
            if model == "Bagging Regressor":
                prediccion = bag_reg.predict(df)
                st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2))) 
            if model == "Regresion lineal":
                df_transformado = transformacion.transform(df)
                prediccion = lin_reg.predict(df_transformado)
                st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "Decision Tree Regressor":
                prediccion = dtr_gs.best_estimator_.predict(df)
                st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "Random Forest Regressor" or model == "Mejor modelo predictivo":
                prediccion = rnd_ft.best_estimator_.predict(df)
                st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "Ada Boost Regressor":
                prediccion = ada_gs.best_estimator_.predict(df)
                st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "Gradient Boosting Regressor":
                prediccion = gbrt.best_estimator_.predict(df)
                st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "PCA con Random Forest Regressor":
                    prediccion = pca_rf.best_estimator_.predict(df)
                    st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
    
    page = st.sidebar.selectbox("Selecciona una página", ("Cliente", "Científicos de Datos"))   # Selector Cliente y DS
        
    if page == "Cliente":
        user_page()
    else:
        data_scientist_page() 

if __name__ == "__main__":
    main()
