import numpy as np
import torch

def load_model(device: str):
    model = ModelHandle(device)
    model = model.load()
    return model

class ModelHandle:
    def __init__(self, device: str):
        self.device = device
        self.model = None
    
    def load(self):
        self.model = torch.load("src/model/artifacts/model.pt", map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        return self.model

