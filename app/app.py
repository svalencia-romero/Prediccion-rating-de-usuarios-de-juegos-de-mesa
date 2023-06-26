import streamlit as st
import pickle
import pandas as pd
# import base64
# Extraer los archivos pickle 

with open("../models/modelo_lineal/trained_lin_reg.pkl", "rb") as li:
    lin_reg = pickle.load(li)

# with open("../models/modelo_lineal/transformacion_polinomio.pkl", "rb") as f:
#     transformacion = pickle.load(f)

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



# Funcion principal de la App

def main():
    # Titulo
    st.title("Modelo Juegos de Mesa")
    # Titulo de sidebar
    st.sidebar.header("Introduzca los parametros necesarios")
    
    option =["Linear Regression","Decision Tree Regressor" ,"Random Forest Regressor","Ada Boost Regressor","Gradient Boosting Regressor","PCA con Random Forest Regressor"]
    model = st.sidebar.selectbox("¿Que modelo quieres probar?",option)

    #funcion para poner los parametros en el sidebar
    
    def user_input_parameters():
        # id = st.sidebar.slider("ID",0,1)
        # year_published = st.sidebar.slider("Year Published",1800,2010)
        min_players = st.sidebar.slider("Min Players",1,10)
        max_players = st.sidebar.slider("Max Players",1,999)
        play_time = st.sidebar.slider("Play Time",1,60000)
        min_age = st.sidebar.slider("Min Age",0,25)
        # users_rated = st.sidebar.slider("Users Rated",30,102214)
        bgg_rank = st.sidebar.slider("BGG Rank",1,20344)
        complejidad_juego = st.sidebar.slider("Complexity Average",0,5)
        owned_users = st.sidebar.slider("Owned Users",1,155312)
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

    st.subheader("User Input Parameters")
    st.subheader(model)
    st.write(df)

    if st.button("RUN"):
        if model == "Linear Regression":
            # df_transformado = transformacion.transform(df)
            prediccion = lin_reg.predict(df)
            st.success(prediccion)
        if model == "Decision Tree Regressor":
            prediccion = dtr_gs.best_estimator_.predict(df)
            st.success(prediccion)
        if model == "Random Forest Regressor":
            prediccion = rnd_ft.best_estimator_.predict(df)
            st.success(prediccion)
        if model == "Ada Boost Regressor":
            prediccion = ada_gs.best_estimator_.predict(df)
            st.success(prediccion)
        if model == "Gradient Boosting Regressor":
            prediccion = gbrt.best_estimator_.predict(df)
            st.success(prediccion)
        if model == "PCA con Random Forest Regressor":
            prediccion = pca_rf.best_estimator_.predict(df)
            st.success(prediccion)

# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('../img/board.jpg')  
# df_conc["Lineal Regression"] = list_lin
# df_conc["Decision Tree Regressor"] = list_dtr
# df_conc["Random Forest"] = list_rdm_fs
# df_conc["Ada Boost Regressor"] = list_ada_gs
# df_conc["Gradient Boosting Regressor"] = list_gbrt
# df_conc["PCA con Random Forest Regressor"] = list_pca_rf

# Llamar a la función para establecer el fondo

if __name__ == "__main__":
    main()
    
