import streamlit as st
import pickle
import pandas as pd

#Extraer los archivos pickle

with open("../models/modelo_lineal/trained_pol_3.pkl", "rb") as li:
    lin_reg = pickle.load(li)

with open("../models/modelo_lineal/transformacion_polinomio.pkl", "rb") as f:
    transformacion = pickle.load(f)

# Funcion

def main():
    # Titulo
    st.title("Modelo Juegos de Mesa")
    # Titulo de sidebar
    st.sidebar.header("Introduzca los parametros necesarios")
    
    #funcion para poner los parametros en el sidebar
    
    def user_input_parameters():
        bgg_rank = st.sidebar.slider("BGG Rank",1,20344)
        Complejidad_juego = st.sidebar.slider("Complexity Average",0,5)
        Mech_role_camp = st.sidebar.slider("Mech_role_camp",0,1)
        Strategy = st.sidebar.slider("Strategy",0,1)
        Wargames = st.sidebar.slider("Wargames",0,1)
        data ={"BGG Rank":bgg_rank,
               "Complexity Average":Complejidad_juego,
               "Mech_role_camp":Mech_role_camp,
               "Strategy":Strategy,
               "Wargames":Wargames,
               }
        features = pd.DataFrame(data,index=[0])
        return features
    
    df = user_input_parameters()
    
    option =["Regresion lineal","Otro modelo"]
    model = st.sidebar.selectbox("¿Que modelo quieres probar?",option)

    st.subheader("User Input Parameters")
    st.subheader(model)
    st.write(df)

    if st.button("RUN"):
        if model == "Regresion lineal":
            df_transformado = transformacion.transform(df)
            prediccion = lin_reg.predict(df_transformado)
            st.success(prediccion)

if __name__ == "__main__":
    main()
