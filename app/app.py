import streamlit as st
import pickle
import pandas as pd
import base64

# Extraer los archivos pickle 

with open("../models/modelo_lineal/trained_pol_3.pkl", "rb") as li:
    lin_reg = pickle.load(li)

with open("../models/modelo_lineal/transformacion_polinomio.pkl", "rb") as f:
    transformacion = pickle.load(f)

with open("../models/arbol_decision/dtr_gs.pkl", "rb") as dtr:
    dtr_gs = pickle.load(dtr)


# Funcion principal de la App

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
    
    option =["Linear Regression","Decision Tree Regressor"]
    model = st.sidebar.selectbox("¿Que modelo quieres probar?",option)

    st.subheader("User Input Parameters")
    st.subheader(model)
    st.write(df)

    if st.button("RUN"):
        if model == "Linear Regression":
            df_transformado = transformacion.transform(df)
            prediccion = lin_reg.predict(df_transformado)
            st.success(prediccion)
        if model == "Decision Tree Regressor":
            prediccion = dtr_gs.best_estimator_.predict(df)
            st.success(prediccion)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('../img/board.jpg')  

# Llamar a la función para establecer el fondo

if __name__ == "__main__":
    main()
    
