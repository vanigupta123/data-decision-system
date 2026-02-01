import random
import fastapi
from metrics.timer import timer
from pipeline.ingest import ingest
from pipeline.preprocess import preprocess
from pipeline.inference import inference
from pipeline.decision import decision
from model.loader import load_model

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(input):
    json = input.load_json()
    if json['request_id'] is None:
        json['request_id'] = random.randint(1, 1000000)
    with timer() as t:
        raw_dict = ingest(json)
        ingest_ms = t.ms
        features = preprocess(raw_dict)
        preprocess_ms = t.ms - ingest_ms
        score, label = inference(features, load_model(json['device']))
        inference_ms = t.ms - (preprocess_ms + ingest_ms)
        label, score, abstained = decision(score, label)
        decision_ms = t.ms - (inference_ms + preprocess_ms + ingest_ms)
        total_ms = t.ms
    return {"request_id": json['request_id'], 
        "decision": {"label": None, "score": None, "abstained": None}, 
        "timing_ms:": {"ingest": ingest_ms, "preprocess": preprocess_ms, "inference": inference_ms, "decision": decision_ms, "total": total_ms}}

@app.get("/health")
def health_check():
    return {"status": "ok"}