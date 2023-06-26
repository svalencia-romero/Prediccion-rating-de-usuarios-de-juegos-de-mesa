import streamlit as st
import pickle
import pandas as pd
import numpy as np


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
# Funcion principal de la App

def main():
       
    # Titulo
    st.title("Modelo Predictivo de Juegos de Mesa")
    st.image("../img/board.jpg", width=650)
    st.write("¡Bienvenido!")
    # Página para científicos de datos
    def user_page():
        st.title("Página para clientes")
                
        def user_input_parameters():
            min_players = st.sidebar.slider("Min Players",1,10)
            max_players = st.sidebar.slider("Max Players",1,20)
            play_time = st.sidebar.slider("Play Time",5,150)
            min_age = st.sidebar.slider("Min Age",0,25)
            bgg_rank = st.sidebar.slider("BGG Rank",1,20344)
            complejidad_juego = st.sidebar.slider("Complexity Average",0,5)
            owned_users = st.sidebar.slider("Owned Users",1,150000)
            mech_not_defined = st.sidebar.slider("Mech Not Defined",0,1)
            mech_acting = st.sidebar.slider("Mech_Acting",0,1)
            mech_action = st.sidebar.slider("Mech_Action",0,1)
            mech_tokens = st.sidebar.slider("Mech_tokens",0,1)
            mech_construcc_farm = st.sidebar.slider("Mech_construcc_farm",0,1)
            mech_roll_thng = st.sidebar.slider("Mech_roll_thng",0,1)
            mech_cards = st.sidebar.slider("Mech_cards",0,1)
            mech_role_camp = st.sidebar.slider("Mech_role_camp",0,1)
            mech_board = st.sidebar.slider("Mech_board",0,1)
            mech_money = st.sidebar.slider("Mech_money",0,1)
            mech_score = st.sidebar.slider("Mech_score",0,1)
            mech_turnbased = st.sidebar.slider("Mech_turnbased",0,1)
            mech_team = st.sidebar.slider("Mech_team",0,1)
            mech_skill = st.sidebar.slider("Mech_skill",0,1)
            mech_solo = st.sidebar.slider("Mech_solo",0,1)
            abstract = st.sidebar.slider("Abstract",0,1)
            children = st.sidebar.slider("Children",0,1)
            customizable = st.sidebar.slider("Customizable",0,1)
            family = st.sidebar.slider("Family",0,1)
            party = st.sidebar.slider("Party",0,1)
            strategy = st.sidebar.slider("Strategy",0,1)
            thematic = st.sidebar.slider("Thematic",0,1)
            wargames = st.sidebar.slider("Wargames",0,1)
            domain_not_defined = st.sidebar.slider("Domain_Not Defined",0,1)      
            
            

            data ={#"ID":id,
                #"Year Published":year_published,
                "Min Players":min_players,
                "Max Players":max_players,
                "Play Time":play_time,
                "Min Age":min_age,
                #"Users Rated":users_rated,
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

        prediccion = rnd_ft.best_estimator_.predict(df)
        st.success("El rating de usuarios es de: " + str(round(prediccion[0], 2)))

        
    def data_scientist_page():
        st.title("Página para Científicos de Datos")
        
        option =["Linear Regression","Decision Tree Regressor" ,"Random Forest Regressor","Ada Boost Regressor","Gradient Boosting Regressor","PCA con Random Forest Regressor"]
        model = st.sidebar.selectbox("¿Que modelo quieres probar?",option)

        show_dataframe = st.checkbox("Mostrar/ocultar Data Frame con errores de cada modelo")
        show_rating = st.checkbox("Mostrar/ocultar rating de usuarios")
        show_money = st.checkbox("Mostrar/ocultar dinero invertido")
                
        def user_input_parameters():
            min_players = st.sidebar.slider("Min Players",1,10)
            max_players = st.sidebar.slider("Max Players",1,20)
            play_time = st.sidebar.slider("Play Time",5,150)
            min_age = st.sidebar.slider("Min Age",0,25)
            bgg_rank = st.sidebar.slider("BGG Rank",1,20344)
            complejidad_juego = st.sidebar.slider("Complexity Average",0,5)
            owned_users = st.sidebar.slider("Owned Users",1,150000)
            mech_not_defined = st.sidebar.slider("Mech Not Defined",0,1)
            mech_acting = st.sidebar.slider("Mech_Acting",0,1)
            mech_action = st.sidebar.slider("Mech_Action",0,1)
            mech_tokens = st.sidebar.slider("Mech_tokens",0,1)
            mech_construcc_farm = st.sidebar.slider("Mech_construcc_farm",0,1)
            mech_roll_thng = st.sidebar.slider("Mech_roll_thng",0,1)
            mech_cards = st.sidebar.slider("Mech_cards",0,1)
            mech_role_camp = st.sidebar.slider("Mech_role_camp",0,1)
            mech_board = st.sidebar.slider("Mech_board",0,1)
            mech_money = st.sidebar.slider("Mech_money",0,1)
            mech_score = st.sidebar.slider("Mech_score",0,1)
            mech_turnbased = st.sidebar.slider("Mech_turnbased",0,1)
            mech_team = st.sidebar.slider("Mech_team",0,1)
            mech_skill = st.sidebar.slider("Mech_skill",0,1)
            mech_solo = st.sidebar.slider("Mech_solo",0,1)
            abstract = st.sidebar.slider("Abstract",0,1)
            children = st.sidebar.slider("Children",0,1)
            customizable = st.sidebar.slider("Customizable",0,1)
            family = st.sidebar.slider("Family",0,1)
            party = st.sidebar.slider("Party",0,1)
            strategy = st.sidebar.slider("Strategy",0,1)
            thematic = st.sidebar.slider("Thematic",0,1)
            wargames = st.sidebar.slider("Wargames",0,1)
            domain_not_defined = st.sidebar.slider("Domain_Not Defined",0,1)      
            
            

            data ={#"ID":id,
                #"Year Published":year_published,
                "Min Players":min_players,
                "Max Players":max_players,
                "Play Time":play_time,
                "Min Age":min_age,
                #"Users Rated":users_rated,
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
        if show_money:
            if df["BGG Rank"].values[0] > 10000:
                cost_per_rank = 5
            else:
                cost_per_rank = 40
            cost_per_user = 0.1
            inv_rank = 20345 - df["BGG Rank"].values[0]
            total_cost = inv_rank * cost_per_rank + df["Owned Users"].values[0] * cost_per_user
            st.write("El coste total de la publicidad de :", round(total_cost, 2),"€")
    
        if show_dataframe:
                st.dataframe(df_errores)      
        if show_rating:
            if model == "Linear Regression":
                    prediccion = lin_reg.predict(df)
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
    page = st.sidebar.selectbox("Selecciona una página", ("Usuarios", "Científicos de Datos"))    
    if page == "Usuarios":
        user_page()
    else:
        data_scientist_page()     

if __name__ == "__main__":
    main()
    
