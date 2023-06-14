import pandas as pd
import numpy as np

df = pd.read_csv("../data/raw/bgg_dataset.csv",sep=";")

""" 
Primero vamos a limpiar todos los datos nulos de las columnas.

0   ID                 
1   Name
2   Year Published 
3   Min Players  
4   Max Players  
5   Play Time 
6   Min Age
7   Users Rated
8   Rating Average
9   BGG Rank 
10  Complexity Average
11  Owned Users
12  Mechanics
13  Domains

"""
# Complexity Average

df["Complexity Average"] = df["Complexity Average"].str.replace(",",".").astype(float) # Cambiamos las comas por puntos y pasamos a float

# Rating Average

df["Rating Average"] = df["Rating Average"].str.replace(",",".").astype(float) # Cambiamos las comas por puntos y pasamos a float

# Year

media_valores_año = int(np.mean(df.loc[df["Year Published"] >= 0, "Year Published"])) # Hacemos la media de los años para dar un valor concreto a los años
df.loc[df["Year Published"] < 1, "Year Published"] = media_valores_año # Miramos que todos los años menores de 1 sean
df.loc[df["Year Published"].isna(), "Year Published"] = media_valores_año #
df["Year Published"] = df["Year Published"].astype(int) # Pasamos a int todos los años

# Domain

df.loc[df["Domains"].isnull(), "Domains"] = "Not Defined" # Definimos como "Not Defined" los valores null del dataset

# Mechanics

df.loc[df["Mechanics"].isnull(), "Mechanics"] = "Not Defined" # Definimos como "Not Defined" los valores null del dataset

# Owned Users

media_valores_owned = int(np.mean(df.loc[df["Owned Users"] >= 0, "Owned Users"]))
df.loc[df["Owned Users"].isna(), "Owned Users"] = media_valores_owned # Definimos los valores null como media de owned por juego.