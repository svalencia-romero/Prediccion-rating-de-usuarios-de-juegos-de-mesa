import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time 
import functions

# Dataframe para pasar a csv en caso de que queramos

df_conc = pd.DataFrame({"Métricas": ["MAE","MAPE","MSE","RMSE", "R2Score"]}).set_index("Métricas")

# Modelo lineal

# Cargamos el modelo entrenado y sus caracteristicas para arreglar con conversion polinómica.

model_path = '../models/modelo_lineal/trained_pol_3.pkl'

loaded_model_lin = functions.load_model(model_path)

# Cargamos data test

df_test = pd.read_csv('../data/test/test.csv')

# Obtener las características (X_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']


# Conversión a polinomica, cargo el pickle hecho

pol_path_mod = '../models/modelo_lineal/transformacion_polinomio.pkl'

poly_feats = functions.load_model(pol_path_mod)

X_test_poly = poly_feats.transform(X_test)

# Realizar las predicciones

predictions = loaded_model_lin.predict(X_test_poly)

# Calcular las métricas de evaluación

mae_lin = mean_absolute_error(y_test, predictions)
mape_lin = mean_absolute_percentage_error(y_test, predictions)
mse_lin = mean_squared_error(y_test, predictions)
rmse_lin = mean_squared_error(y_test, predictions, squared=False)
r2_lin = r2_score(y_test, predictions)

list_lin = [round(mae_lin,4),round(mape_lin,4),round(mse_lin,4),round(rmse_lin,4),round(r2_lin,4)]

# Imprimir las métricas

print("Métricas del modelo lineal","\n")
print("Mean Absolute Error (MAE):", round(mae_lin,4))
print("Mean Absolute Percentage Error (MAPE):", round(mape_lin,4))
print("Mean Squared Error (MSE):", round(mse_lin,4))
print("Root Mean Squared Error (RMSE):", round(rmse_lin,4))
print("R-squared (R2) Score:", round(r2_lin,4),"\n")


# Modelo Arbol decisión

# Carga de modelo
model_path = '../models/arbol_decision/dtr_gs.pkl'

loaded_model_dtr_gs = functions.load_model(model_path)

# Obtener el mejor modelo entrenado

y_pred_dtr = loaded_model_dtr_gs.best_estimator_.predict(X_test)

mae_dtr = mean_absolute_error(y_test, y_pred_dtr)
mape_dtr = mean_absolute_percentage_error(y_test, y_pred_dtr)
mse_dtr = mean_squared_error(y_test, y_pred_dtr)
rmse_dtr = mean_squared_error(y_test, y_pred_dtr, squared=False)
r2_dtr = r2_score(y_test, y_pred_dtr)
print("Métricas del modelo arbol de decisión","\n")
print("Mean Absolute Error (MAE):", round(mae_dtr,4))
print("Mean Absolute Percentage Error (MAPE):", round(mape_dtr,4))
print("Mean Squared Error (MSE):", round(mse_dtr,4))
print("Root Mean Squared Error (RMSE):", round(rmse_dtr,4))
print("R-squared (R2) Score:", round(r2_dtr,4),"\n")

list_dtr = [round(mae_dtr,4),round(mape_dtr,4),round(mse_dtr,4),round(rmse_dtr,4),round(r2_dtr,4)]

print("Evaluación Finalizada")
print()
pregunta = input("¿Quieres un csv con los resultados obtenidos? (S/N) ")
if pregunta == "S" or pregunta == "s":
    df_conc["Lineal Regression"] = list_lin
    df_conc["Decision Tree Regressor"] = list_dtr
    df_conc.to_csv("../data/processed/analisis_metricas.csv")
    print("CSV creado con éxito")

print("Presiona ENTER para salir...")
input()
