import random
from fastapi import APIRouter, Depends
from metrics.timer import timer
from pipeline.ingest import PredictRequest, ingest
from pipeline.preprocess import preprocess
from pipeline.inference import inference
from pipeline.decision import decide
from app.dependencies import get_model
from app.main import main
from utils.logging import Counters, log_request

router = APIRouter()
counters = Counters()

@router.get("/")
def read_root():
    return {"message": "Hello, World!"}

@router.post("/predict")
def predict(request: PredictRequest, model: Depends(get_model)):
    timings = {}
    try:
        with timer() as t:
            raw_dict = ingest(request)
            ingest_ms = t.ms # 1
            features = preprocess(raw_dict)
            preprocess_ms = t.ms - ingest_ms
            pred = inference(features, get_model())
            inference_ms = t.ms - preprocess_ms
            dec = decide(pred)
            decision_ms = t.ms - inference_ms
            total_ms = t.ms
        timings = {"ingest": ingest_ms, "preprocess": preprocess_ms, "inference": inference_ms, "decision": decision_ms, "total": total_ms}
        log_request(request_id=raw_dict['request_id'], timings_ms=timings, counters=counters, status="success", decision=dec)
        return {"request_id": raw_dict['request_id'], 
        "decision": {"label": dec.label, "score": dec.score, "abstained": dec.abstained}, 
        "timing_ms:": timings}
    except Exception as e:
        # TODO: consider adding "extra" field contents
        log_request(request_id=raw_dict['request_id'], timings_ms=timings, counters=counters, status="failure", decision=dec, error=str(e), error_type=type(e).__name__)
        raise

@router.get("/health")
def health_check():
    return {"status": "ok"}