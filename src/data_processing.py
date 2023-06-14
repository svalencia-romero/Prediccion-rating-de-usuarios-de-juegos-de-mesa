import pandas as pd
import numpy as np

df = pd.read_csv("../data/raw/bgg_dataset.csv",sep=";")

""" 
Primero vamos a limpiar todos los datos ya que había bastantes cosas a tener en cuenta.
"""
df["Complexity Average"] = df["Complexity Average"].str.replace(",",".").astype(float) # Cambiamos las comas por puntos y pasamos a float

df["Rating Average"] = df["Rating Average"].str.replace(",",".").astype(float) # Cambiamos las comas por puntos y pasamos a float


media_valores_año = int(np.mean(df.loc[df["Year Published"] >= 0, "Year Published"])) #Encontramos todos los valores menores de 
df.loc[df["Year Published"] < 1, "Year Published"] = media_valores_año
df.loc[df["Year Published"].isna(), "Year Published"] = media_valores_año
df["Year Published"] = df["Year Published"].astype(int)
df.loc[df["Domains"].isnull(), "Domains"] = "Not Defined" # Definimos como "Not Defined" los valores null del dataset