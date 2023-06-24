import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score 
import functions

# Dataframe para pasar a csv en caso de que queramos

df_conc = pd.DataFrame({"Métricas": ["MAE","MAPE","MSE","RMSE", "R2Score"]}).set_index("Métricas")

# Cargamos data test

df_test = pd.read_csv('../data/test/test.csv')

# Obtener las características (X_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']

# Modelo lineal

# Cargamos el modelo entrenado y sus caracteristicas para arreglar con conversion polinómica.

model_path = '../models/modelo_lineal/trained_pol_3.pkl'

loaded_model_lin = functions.load_model(model_path)

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

# Cargamos data test

df_test = pd.read_csv('../data/test/test.csv')
df_train = pd.read_csv('../data/train/train.csv')

# Obtener las características (X_test) y las etiquetas (y_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']

X_train = df_train.drop('Rating Average', axis=1)
y_train = df_train['Rating Average']

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

# Modelo Random Forest

# Carga de modelo
model_path = '../models/random_forest/rnd_ft.pkl'

loaded_model_rdm_fs = functions.load_model(model_path)

# Cargamos data test

df_test = pd.read_csv('../data/test/test.csv')
df_train = pd.read_csv('../data/train/train.csv')

# Obtener las características (X_test) y las etiquetas (y_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']

X_train = df_train.drop('Rating Average', axis=1)
y_train = df_train['Rating Average']

# Obtener el mejor modelo entrenado

y_pred_rdm_fs = loaded_model_rdm_fs.best_estimator_.predict(X_test)

mae_rdm_fs = mean_absolute_error(y_test, y_pred_rdm_fs)
mape_rdm_fs = mean_absolute_percentage_error(y_test, y_pred_rdm_fs)
mse_rdm_fs = mean_squared_error(y_test, y_pred_rdm_fs)
rmse_rdm_fs = mean_squared_error(y_test, y_pred_rdm_fs, squared=False)
r2_rdm_fs = r2_score(y_test, y_pred_rdm_fs)
print("Métricas del modelo Random Forest","\n")
print("Mean Absolute Error (MAE):", round(mae_rdm_fs,4))
print("Mean Absolute Percentage Error (MAPE):", round(mape_rdm_fs,4))
print("Mean Squared Error (MSE):", round(mse_rdm_fs,4))
print("Root Mean Squared Error (RMSE):", round(rmse_rdm_fs,4))
print("R-squared (R2) Score:", round(r2_rdm_fs,4),"\n")

list_rdm_fs = [round(mae_rdm_fs,4),round(mape_rdm_fs,4),round(mse_rdm_fs,4),round(rmse_rdm_fs,4),round(r2_rdm_fs,4)]

# Modelo Ada Boost

# Carga de modelo
model_path = '../models/ada_gs/ada_gs.pkl'

loaded_model_ada_gs = functions.load_model(model_path)

# Cargamos data test

df_test = pd.read_csv('../data/test/test.csv')
df_train = pd.read_csv('../data/train/train.csv')

# Obtener las características (X_test) y las etiquetas (y_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']

X_train = df_train.drop('Rating Average', axis=1)
y_train = df_train['Rating Average']

# Obtener el mejor modelo entrenado

y_pred_ada_gs = loaded_model_ada_gs.best_estimator_.predict(X_test)

mae_ada_gs = mean_absolute_error(y_test, y_pred_ada_gs)
mape_ada_gs = mean_absolute_percentage_error(y_test, y_pred_ada_gs)
mse_ada_gs = mean_squared_error(y_test, y_pred_ada_gs)
rmse_ada_gs = mean_squared_error(y_test, y_pred_ada_gs, squared=False)
r2_ada_gs = r2_score(y_test, y_pred_ada_gs)
print("Métricas del modelo Ada Boost","\n")
print("Mean Absolute Error (MAE):", round(mae_ada_gs,4))
print("Mean Absolute Percentage Error (MAPE):", round(mape_ada_gs,4))
print("Mean Squared Error (MSE):", round(mse_ada_gs,4))
print("Root Mean Squared Error (RMSE):", round(rmse_ada_gs,4))
print("R-squared (R2) Score:", round(r2_ada_gs,4),"\n")

list_ada_gs = [round(mae_ada_gs,4),round(mape_ada_gs,4),round(mse_ada_gs,4),round(rmse_ada_gs,4),round(r2_ada_gs,4)]
print("Evaluación Finalizada")
print()
pregunta = input("¿Quieres un csv con los resultados obtenidos? (S/N) ")
if pregunta == "S" or pregunta == "s":
    df_conc["Lineal Regression"] = list_lin
    df_conc["Decision Tree Regressor"] = list_dtr
    df_conc["Random Forest"] = list_rdm_fs
    df_conc["Ada Boost Regressor"] = list_ada_gs
    df_conc.to_csv("../data/processed/analisis_metricas.csv")
    print("CSV creado con éxito")

print("Presiona ENTER para salir...")
input()
