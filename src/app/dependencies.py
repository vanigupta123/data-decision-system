from model.loader import load_model
import torch
from app import app

def get_model():
    model = app.state.model
    return model
