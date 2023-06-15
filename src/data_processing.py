import pandas as pd
import numpy as np
np.random.seed(22)

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
# ID

valores_nulos = df["ID"].isnull() # Vemos los valores nulos
valores_no_nulos = df["ID"].dropna().unique() # Sacamos los valores que no son nulos
valores_aleatorios = np.random.choice(valores_no_nulos, size=valores_nulos.sum(), replace=False) # Hacemos valores aleatorios en una variable
df.loc[valores_nulos, "ID"] = valores_aleatorios # Cambiamos valores nulos y sustituimos por valores aleatorios 

# Name la dejamos igual.

# Year Published

media_valores_año = int(np.mean(df.loc[df["Year Published"] >= 0, "Year Published"])) # Hacemos la media de los años para dar un valor concreto a los años
df.loc[df["Year Published"] < 1, "Year Published"] = media_valores_año # Miramos que todos los años menores de 1 sean
df.loc[df["Year Published"].isna(), "Year Published"] = media_valores_año #
df["Year Published"] = df["Year Published"].astype(int) # Pasamos a int todos los años

# Min Players la dejamos igual.

# Max Players la dejamos igual.

# Play Time la dejamos igual.

# Min Age la dejamos igual.

# Users Rated la dejamos igual.

# Rating Average

df["Rating Average"] = df["Rating Average"].str.replace(",",".").astype(float) # Cambiamos las comas por puntos y pasamos a float

# BGG Rank la dejamos igual.

# Complexity Average

df["Complexity Average"] = df["Complexity Average"].str.replace(",",".").astype(float) # Cambiamos las comas por puntos y pasamos a float

# Owned Users

media_valores_owned = int(np.mean(df.loc[df["Owned Users"] >= 0, "Owned Users"]))
df.loc[df["Owned Users"].isna(), "Owned Users"] = media_valores_owned # Definimos los valores null como media de owned por juego.
df["Owned Users"] = df["Owned Users"].astype(int)

# Mechanics

df.loc[df["Mechanics"].isnull(), "Mechanics"] = "Not Defined" # Definimos como "Not Defined" los valores null del dataset

# Domain

df.loc[df["Domains"].isnull(), "Domains"] = "Not Defined" # Definimos como "Not Defined" los valores null del dataset

df.to_csv("../data/processed/bgg_proc.csv")
