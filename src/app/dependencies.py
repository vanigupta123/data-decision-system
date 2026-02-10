from model.loader import load_model
from fastapi import Request

def get_model(request: Request):
    return request.app.state.model
