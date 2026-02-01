from pydantic import RawInput

def ingest(input_json: dict):
    return RawInput(**input_json)
    