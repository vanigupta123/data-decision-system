import numpy as np
from pipeline.preprocess import preprocess
from model.loader import ModelHandle

def inference(features: np.array, model: ModelHandle):
    return score, label