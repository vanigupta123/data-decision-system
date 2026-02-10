import numpy as np
from pipeline.preprocess import Features, preprocess
from model.loader import ModelHandle
import torch

class Prediction:
    def __init__(self, score, label):
        self.score = score
        self.label = label

def inference(features: Features, model: ModelHandle, threshold: float = 0.5):
    x = features.x
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() != 2:
        raise ValueError(f"expected x of dimensions 1 or 2, got {x.dim()}")
    
    try:
        device = next(model.parameters()).device
        x = x.to(device)
    except:
        pass

    with torch.no_grad():
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
    
    if not torch.is_tensor(y): # this is literally never getting hit but just in case??
        raise ValueError("output is not a tensor")

    y = y.detach().float().cpu()
    if y.numel() == 1:
        score = float(y.view(-1)[0])
    else:
        score = float(y.view(-1)[-1]) # TODO: this can be more explicit later but for now we assume class 1

    label = 1 if score >= threshold else 0
    return Prediction(score = score, label=label)

