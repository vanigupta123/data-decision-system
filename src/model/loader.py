import numpy as np

def load_model(device: str):
    model = ModelHandle(device)
    return model

class ModelHandle:
    def __init__(self, device: str):
        self.device = device

