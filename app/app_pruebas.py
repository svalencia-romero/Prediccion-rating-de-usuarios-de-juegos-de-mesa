import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 
# Carga de configuraciones de los modelos

with open("../models/arbol_decision/model_config_dtr.yaml", "r") as file:
    model_cfg_dtr = yaml.safe_load(file)

with open("../models/gbrt/model_config_gbrt.yaml", "r") as file:
    model_cfg_gbrt = yaml.safe_load(file)

with open("../models/pca_rf/model_config_pca_rf.yaml", "r") as file:
    model_cfg_pca_rf = yaml.safe_load(file)

with open("../models/random_forest/model_config_rnd_ft.yaml", "r") as file:
    model_cfg_rnd_ft = yaml.safe_load(file)

with open("../models/ada_gs/model_config_ada_gs.yaml", "r") as file:
    model_cfg_ada_gs = yaml.safe_load(file)


with open("../models/modelo_lineal/trained_lin_reg.pkl", "rb") as li:
    lin_reg = pickle.load(li)

with open("../models/arbol_decision/dtr_gs.pkl", "rb") as dtr:
    dtr_gs = pickle.load(dtr)

with open("../models/random_forest/rnd_ft.pkl", "rb") as dtr:
    rnd_ft = pickle.load(dtr)    

with open("../models/ada_gs/ada_gs.pkl", "rb") as ada:
    ada_gs = pickle.load(ada)

with open("../models/gbrt/gbrt.pkl", "rb") as gbrt:
    gbrt = pickle.load(gbrt)

with open("../models/pca_rf/pca_rf.pkl", "rb") as pca:
    pca_rf = pickle.load(pca)

df_errores = pd.read_csv("../data/processed/analisis_metricas.csv",index_col="Métricas")

st.set_page_config(layout="wide")

# Funcion principal del dataframe que vamos modificando

def user_input_parameters():
            min_players = st.sidebar.slider("Mínimo número de jugadores",1,10)
            max_players = st.sidebar.slider("Máximo número de jugadores",1,20)
            play_time = st.sidebar.slider("Tiempo de juego aproximado",5,150)
            min_age = st.sidebar.slider("Edad mínima",0,25)
            bgg_rank = st.sidebar.slider("Rango aproximado en BGG que crees que puede tener",1,20344,step=1)
            complejidad_juego = st.sidebar.slider("Grado de complejidad",0,5)
            owned_users = st.sidebar.slider("Usuarios que aproximadamente tienen el juego",1,150000,step=1)
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


# Funcion principal de la App


def main():
       
    # Titulo
    st.title("Modelo Predictivo de Juegos de Mesa")
    st.image("../img/board.jpg", width=650)
    st.write("¡Bienvenido!")
    # Página para científicos de datos
    def user_page():
        st.title("Página para cliente final")
        st.write()
        show_money = st.checkbox("En este apartado podrás comprobar diferentes cuestiones economicas importantes sobre tu juego.")
        show_rating = st.checkbox("Prediccion de rating de usuarios")
        df = user_input_parameters()                       
        
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
            coste_manuf_game = st.number_input("¿Cuanto dinero tienes previsto invertir en fabricar tu juego?",value=2000,step=1)
            gan_per_game = (coste_de_juego * 0.80) - cost_per_rank # Lo que se lleva el cliente por juegos.
            gan_game_overall = (df["Owned Users"].values[0] * gan_per_game)  - coste_manuf_game
            inv_rank = 20345 - df["BGG Rank"].values[0]
            total_cost = inv_rank * cost_per_rank + df["Owned Users"].values[0] * cost_per_user_tienda
            st.write("Dinero obtenido por cada juego:", round(gan_per_game, 2),"€")
            st.write("Total del dinero por todos los juegos:", round(gan_game_overall, 2),"€")
            st.write("Estimación realizada si tiendas reciben el 20 % de las ventas de cada juego")
            st.write("El coste aproximado total de publicidad es de :", round(total_cost, 2),"€")

    def data_scientist_page():
        st.title("Página para Científicos de Datos")
        
        option =["Linear Regression","Decision Tree Regressor" ,"Random Forest Regressor","Ada Boost Regressor","Gradient Boosting Regressor","PCA con Random Forest Regressor"]
        model = st.sidebar.selectbox("¿Que modelo quieres probar?",option)

        show_dataframe = st.checkbox("Mostrar/ocultar Data Frame con errores de cada modelo")
        show_rating = st.checkbox("Mostrar/ocultar rating de usuarios")
        # show_grafs = st.checkbox("Mostrar/ocultar graficas")
        show_config = st.checkbox("Mostrar/ocultar configuraciones de modelos")
        
        # show_grafs
                
        df = user_input_parameters()
        
        cfg_options = ["Decision Tree Regressor",
            "Ada Boost Regressor",
            "Gradient Boosting Regressor",
            "PCA con Random Forest Regressor",
            "Random Forest Regressor"
            ]

        cfg = st.selectbox("Seleccionar configuración", cfg_options)

        if show_config:
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

            
        if show_dataframe:
                st.dataframe(df_errores)  

        if show_rating:
            if model == "Linear Regression":
                    prediccion = lin_reg.predict(df)
                    st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "Decision Tree Regressor":
                    prediccion = dtr_gs.best_estimator_.predict(df)
                    st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))
            if model == "Random Forest Regressor":
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

    page = st.sidebar.selectbox("Selecciona una página", ("Usuarios", "Científicos de Datos"))    
    if page == "Usuarios":
        user_page()
    else:
        data_scientist_page()     

if __name__ == "__main__":
    main()
    
