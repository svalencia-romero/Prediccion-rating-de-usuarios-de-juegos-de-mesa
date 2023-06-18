import pandas as pd
import numpy as np
np.random.seed(22)
# import os
# print(os.getcwd())

df = pd.read_csv("Prediccion-sobre-juegos-de-mesa/data/raw/bgg_dataset.csv",sep=";")

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

# Conversión de dataset procesado a csv con los datos limpios sin tocar variables Mechanics y Domain

df.to_csv("Prediccion-sobre-juegos-de-mesa/data/processed/bgg_proc_clean.csv",index=False)

# Procesamiento de Mechanics

df_descriptions = df['Mechanics'].str.get_dummies(', ')
df = pd.concat([df, df_descriptions], axis=1)

df["Mech_Acting"] = df[["Communication Limits","Alliances","Acting","Player Judge","Role Playing","Roles with Asymmetric Information","Storytelling", "Traitor Game","Catch the Leader","Follow","Force Commitment","Hidden Roles","Negotiation","Voting"]].any(axis='columns').astype(int)
df["Mech_Action"] = df[["Static Capture","Sudden Death Ending","Simultaneous Action Selection","Matching","Cube Tower","Bias","Real-Time","Push Your Luck","Action Drafting", "Action Points","Action Queue","Action Retrieval","Action Timer","Action/Event","Critical Hits and Failures","Hot Potato","I Cut You Choose","Interrupts","Kill Steal","Ladder Climbing","Line Drawing","Paper-and-Pencil","Race","Singing","Take That","Three Dimensional Movement","Track Movement","Measurement Movement"]].any(axis='columns').astype(int)
df["Mech_tokens"] = df[["Physical Removal","Advantage Token", "Passed Action Token"]].any(axis='columns').astype(int)
df["Mech_construcc_farm"] = df[["Tech Trees / Tech Tracks","Resource to Move","Automatic Resource Growth","Random Production","Mancala","Network and Route Building","Ownership","Pattern Building","Pick-up and Deliver"]].any(axis='columns').astype(int)
df["Mech_roll_thng"] = df[["Variable Set-up","Worker Placement with Dice Workers","Roll / Spin and Move","Flicking","Re-rolling and Locking","Rondel","Dice Rolling","Different Dice Movement"]].any(axis='columns').astype(int)
df["Mech_cards"] = df[["Trick-taking","Melding and Splaying","Drafting","Set Collection","Move Through Deck","Hand Management","Deck Construction","Deck Bag and Pool Building","Campaign / Battle Card Driven","Card Drafting","Card Play Conflict Resolution","Chaining","Command Cards"]].any(axis='columns').astype(int)
df["Mech_role_camp"] = df[["Variable Player Powers","Worker Placement","Stat Check Resolution","Simulation","Scenario / Mission / Campaign Game","Ratio / Combat Results Table","Narrative Choice / Paragraph","Multiple Maps","Legacy Game","Finale Ending","Events","End Game Bonuses","Enclosure","Elapsed Real Time Ending","Different Worker Types","Die Icon Resolution"]].any(axis='columns').astype(int)
df["Mech_board"] = df[["Tile Placement","Square Grid","Slide/Push","Line of Sight","Zone of Control","Secret Unit Deployment","Point to Point Movement","Pieces as Map","Moving Multiple Units","Movement Template","Movement Points","Modular Board","Minimap Resolution","Map Reduction","Map Deformation","Map Addition","Layering","Impulse Movement","Hidden Movement","Hexagon Grid","Grid Movement","Grid Coverage","Crayon Rail System","Connections","Area Movement","Area Majority / Influence","Area-Impulse"]].any(axis='columns').astype(int)
df["Mech_money"] = df[["Trading","Stock Holding","Stacking and Balancing","Selection Order Bid","Predictive Bid","Order Counters","Multiple-Lot Auction","Market","Loans","Investment","Increase Value of Unchosen Resources","Income","Delayed Purchase","Contracts","Closed Economy Auction","Constrained Bidding","Commodity Speculation","Bribery","Bingo","Betting and Bluffing","Auction/Bidding","Auction: Dexterity","Auction: Dutch","Auction: Dutch Priority","Auction: English","Auction: Fixed Placement","Auction: Once Around","Auction: Sealed Bid","Auction: Turn Order Until Pass"]].any(axis='columns').astype(int)
df["Mech_score"] = df[["Victory Points as a Resource","Player Elimination","Highest-Lowest Scoring","Hidden Victory Points","Score-and-Reset Game"]].any(axis='columns').astype(int)
df["Mech_turnbased"] = df[["Variable Phase Order","Turn Order: Stat-Based","Turn Order: Role Order","Turn Order: Random","Turn Order: Progressive","Turn Order: Pass Order","Turn Order: Claim Action","Turn Order: Auction","Relative Movement","Programmed Movement","Pattern Recognition","Pattern Movement","Lose a Turn","Chit-Pull System"]].any(axis='columns').astype(int)
df["Mech_team"] = df[["Tug of War","Prisoner's Dilemma","Team-Based Game","Semi-Cooperative Game","Cooperative Game"]].any(axis='columns').astype(int)
df["Mech_skill"] = df[["Time Track","Targeted Clues","Induction","Deduction","Speed Matching","Rock-Paper-Scissors","Once-Per-Game Abilities","Memory"]].any(axis='columns').astype(int)
df["Mech_solo"] = df[["King of the Hill","Single Loser Game","Solo / Solitaire Game"]].any(axis='columns').astype(int)
df = df.rename(columns={"Not Defined":"Mech Not Defined"})
df.drop(["King of the Hill","Communication Limits","Alliances","Acting","Player Judge","Role Playing","Roles with Asymmetric Information","Storytelling", "Traitor Game","Catch the Leader","Follow","Force Commitment","Hidden Roles","Negotiation","Voting","Static Capture","Sudden Death Ending","Simultaneous Action Selection","Matching","Cube Tower","Bias","Real-Time","Push Your Luck","Action Drafting", "Action Points","Action Queue","Action Retrieval","Action Timer","Action/Event","Critical Hits and Failures","Hot Potato","I Cut You Choose","Interrupts","Kill Steal","Ladder Climbing","Line Drawing","Paper-and-Pencil","Race","Singing","Take That","Three Dimensional Movement","Track Movement","Measurement Movement","Physical Removal","Advantage Token", "Passed Action Token","Tech Trees / Tech Tracks","Resource to Move","Automatic Resource Growth","Random Production","Mancala","Network and Route Building","Ownership","Pattern Building","Pick-up and Deliver","Variable Set-up","Worker Placement with Dice Workers","Roll / Spin and Move","Flicking","Re-rolling and Locking","Rondel","Dice Rolling","Different Dice Movement","Trick-taking","Melding and Splaying","Drafting","Set Collection","Move Through Deck","Hand Management","Deck Construction","Deck Bag and Pool Building","Campaign / Battle Card Driven","Card Drafting","Card Play Conflict Resolution","Chaining","Command Cards","Variable Player Powers","Worker Placement","Stat Check Resolution","Simulation","Scenario / Mission / Campaign Game","Ratio / Combat Results Table","Narrative Choice / Paragraph","Multiple Maps","Legacy Game","Finale Ending","Events","End Game Bonuses","Enclosure","Elapsed Real Time Ending","Different Worker Types","Die Icon Resolution","Tile Placement","Square Grid","Slide/Push","Line of Sight","Zone of Control","Secret Unit Deployment","Point to Point Movement","Pieces as Map","Moving Multiple Units","Movement Template","Movement Points","Modular Board","Minimap Resolution","Map Reduction","Map Deformation","Map Addition","Layering","Impulse Movement","Hidden Movement","Hexagon Grid","Grid Movement","Grid Coverage","Crayon Rail System","Connections","Area Movement","Area Majority / Influence","Area-Impulse","Trading","Stock Holding","Stacking and Balancing","Selection Order Bid","Predictive Bid","Order Counters","Multiple-Lot Auction","Market","Loans","Investment","Increase Value of Unchosen Resources","Income","Delayed Purchase","Contracts","Closed Economy Auction","Constrained Bidding","Commodity Speculation","Bribery","Bingo","Betting and Bluffing","Auction/Bidding","Auction: Dexterity","Auction: Dutch","Auction: Dutch Priority","Auction: English","Auction: Fixed Placement","Auction: Once Around","Auction: Sealed Bid","Auction: Turn Order Until Pass","Victory Points as a Resource","Player Elimination","Highest-Lowest Scoring","Hidden Victory Points","Score-and-Reset Game","Variable Phase Order","Turn Order: Stat-Based","Turn Order: Role Order","Turn Order: Random","Turn Order: Progressive","Turn Order: Pass Order","Turn Order: Claim Action","Turn Order: Auction","Relative Movement","Programmed Movement","Pattern Recognition","Pattern Movement","Lose a Turn","Chit-Pull System","Tug of War","Prisoner's Dilemma","Team-Based Game","Semi-Cooperative Game","Cooperative Game","Time Track","Targeted Clues","Induction","Deduction","Speed Matching","Rock-Paper-Scissors","Once-Per-Game Abilities","Memory","Single Loser Game","Solo / Solitaire Game"],inplace=True,axis=1)

# Procesamiento de Domain

df = df.join(df['Domains'].str.split(expand=True))
df = pd.get_dummies(df, columns=[0,1,2,3,4,5])

df["Children"] = df[["0_Children's", "2_Children's"]].any(axis='columns').astype(int)
df["Customizable"] = df[["2_Customizable", "0_Customizable"]].any(axis='columns').astype(int)
df["Family"] = df[["2_Family", "0_Family"]].any(axis='columns').astype(int)
df["Party"] = df[["0_Party", "2_Party","4_Party"]].any(axis='columns').astype(int)
df["Strategy"] = df[["0_Strategy", "2_Strategy"]].any(axis='columns').astype(int)
df["Thematic"] = df[["0_Thematic", "2_Thematic","4_Thematic"]].any(axis='columns').astype(int)
df["Wargames"] = df[["0_Wargames", "2_Wargames","4_Wargames"]].any(axis='columns').astype(int)
df["Domain_Not Defined"] = df[["0_Not","1_Defined"]].any(axis='columns').astype(int)
df = df.rename(columns={"0_Abstract":"Abstract"})
df["Abstract"].astype(int)
df.drop(["0_Children's", '0_Customizable', '0_Family', '0_Not',
       '0_Party', '0_Strategy', '0_Thematic', '0_Wargames', '1_Defined',
       '1_Games', '1_Games,', "2_Children's", '2_Customizable', '2_Family',
       '2_Party', '2_Strategy', '2_Thematic', '2_Wargames', '3_Games',
       '3_Games,', '4_Party', '4_Thematic', '4_Wargames', '5_Games'],inplace=True,axis=1)



df.to_csv("Prediccion-sobre-juegos-de-mesa/data/processed/bgg_proc_feat.csv",index=False)
