from typing import Any, List, Dict
from pydantic.dataclasses import dataclass
import numpy as np
import torch

default_vals = {"temperature": 0.0}
norm_constants = {"temperature": 0.0}
std = {"temperature": 1.0} # TODO

@dataclass(frozen=True)
class Features:
    x: torch.Tensor
    features: List[str]
    missing_features: List[str]

def preprocess(raw_input):
    values = list[float] = []
    missing = list[str] = []
    for fname in default_vals.keys():
        val = getattr(raw_input.features, fname)
        if val is None:
            val = default_vals[fname]
            missing.append(fname)
        
        try:
            v = float(val)
        except Exception as e:
            raise ValueError(f"feature {fname} could not be cast to float: {val}")

        v = (v - norm_constants[fname]) / std[fname]
        values.append(v)

    x = torch.tensor(values, dtype=torch.float32)
    return Features(x=x, features=default_vals.keys(), missing_features=missing)
