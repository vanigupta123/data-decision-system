import math
from random import random
from typing import Optional
from pydantic import Field, BaseModel, field_validator

class RunConfigs(BaseModel):
    missing_rate: float = Field(0.0, ge=0.0, le=0.0)
    delay_steps: int = Field(0, ge=0)
    noise_std: float = Field(0.0, ge=0.0)
    seed: Optional[int] = None

class FeaturesPredict(BaseModel):
    temperature: float

    @field_validator("*")
    @classmethod
    def check_values(cls, v):
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise ValueError("feature input must be numeric")
        if v != v or v == math.inf or v == -math.inf:
            raise ValueError("feature input contains NaN")
        return v

class PredictRequest(BaseModel):
    request_id: Optional[str] = None
    features: FeaturesPredict
    run_config: Optional[RunConfigs] = RunConfigs()

def ingest(input: dict):
    try:
        request = PredictRequest(**input)
        if request.request_id is None:
            request.request_id = random.randint(1, 1000000)
    except Exception as e:
        raise ValueError(f"invalid request: {e}")
    return request
