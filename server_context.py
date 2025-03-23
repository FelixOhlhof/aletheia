import os
from typing import Dict
from aletheialib_ext.models import NN

class ServerContext:
    TMP_DIR=os.path.join(os.path.dirname(__file__), "temp")
    b  = ""
    
    def __init__(self, models_path=None, load_models_lazy=True):
        print("initializing server context...")
        if models_path == None:
            models_path = "aletheia-models"
        self.models : Dict[str, NN] = self.load_models(models_path, load_models_lazy)

    def load_models(self, models_path, load_models_lazy):
        models = {}

        for filename in os.listdir(models_path):
            if filename.endswith(".h5"):
                model_path = os.path.join(models_path, filename)
                models[filename[:-3]] = NN(model_path, load_models_lazy)
        return models