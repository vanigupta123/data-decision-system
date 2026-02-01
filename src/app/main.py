from model.loader import load_model
import torch
from fastapi import FastAPI

def main():
    app = FastAPI()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    app.state.model = model
    return app