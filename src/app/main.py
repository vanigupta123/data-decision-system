from model.loader import load_model
import torch
from fastapi import FastAPI
from src.app.routes import router

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
def _startup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = load_model(device)
    return app