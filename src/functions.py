import yaml
import pickle

# Funcion de carga de modelo .pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Función de carga de caracteristicas de modelo

def load_config(model_path):
    with open(model_path, 'r') as c:
        model = yaml.safe_load(c)
    return model
  