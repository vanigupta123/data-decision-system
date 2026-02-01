from pydantic import RawInput
from pipeline.ingest import ingest
import numpy as np

def preprocess(raw_input: RawInput):
    return np.array(raw_input.input)