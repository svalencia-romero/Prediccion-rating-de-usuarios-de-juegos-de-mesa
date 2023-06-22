import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time 
import yaml


# Llamamos al modelo concreto de ML que estamos utilizando
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargamos el modelo entrenado y sus caracteristicas para arreglar con conversion polinómica.

with open('../models/model_config.yaml', 'r') as file:
    model_config = yaml.safe_load(file)

model_path = '../models/trained_pol_3.pkl'

loaded_model = load_model(model_path)

# Cargamos data test

df_test = pd.read_csv('../data/test/test.csv')
df_train = pd.read_csv('../data/train/train.csv')

# Obtener las características (X_test) y las etiquetas (y_test)

X_test = df_test.drop('Rating Average', axis=1)
y_test = df_test['Rating Average']

X_train = df_train.drop('Rating Average', axis=1)
y_train = df_train['Rating Average']

# Conversión a polinomica, cargo el pickle hecho

pol_path_mod = '../models/transformacion_polinomio.pkl'

poly_feats = load_model(pol_path_mod)

X_test_poly = poly_feats.transform(X_test)

# Realizar las predicciones

predictions = loaded_model.predict(X_test_poly)

# Calcular las métricas de evaluación

mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

# Imprimir las métricas

print("Mean Absolute Error (MAE):", round(mae,4))
print("Mean Absolute Percentage Error (MAPE):", round(mape,4))
print("Mean Squared Error (MSE):", round(mse,4))
print("Root Mean Squared Error (RMSE):", round(rmse,4))
print("R-squared (R2) Score:", round(r2,4))

time.sleep(5)
