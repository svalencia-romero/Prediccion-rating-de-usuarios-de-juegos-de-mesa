import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import yaml

# Llamamos al modelo concreto de ML que estamos utilizando
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model



model_path = '../models/trained_pol_3.pkl'

loaded_model = load_model(model_path)

# Convertimos a polinomio
poly_reg = PolynomialFeatures(degree = 3)
poly_reg.fit("../data/train/X_train.csv")
X_train_poly = poly_reg.transform("../data/test/X_train.csv")
X_test_poly = poly_reg.transform("../data/test/X_test.csv")
train_prediction = pol_reg.predict(X_train_poly)

print(pol_reg.score(X_train_poly, "../data/train/y_train.csv"))
print("MAE train", mean_absolute_error("../data/train/y_train.csv", train_prediction))
print("MSE train", mean_squared_error("../data/train/y_train.csv", train_prediction))
