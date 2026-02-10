from src.pipeline.inference import Prediction

def decide(pred: Prediction):
    return Decision(pred)

class Decision:
    def __init__(self, pred: Prediction):
        self.label = pred.label
        self.score = pred.score
        self.abstained = False # TODO: change based on cost/instability/latency
        self.reason = None