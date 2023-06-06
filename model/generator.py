import numpy as np
from model.utils import mean_embeddings, similarity_score


class Summarizer:
    def __init__(self, method, length):
        self.method = method
        self.length = length

    def detect_keyframes(self, frames, scores):
        if self.method == "max":
            indices = np.argpartition(scores, -self.length)[-self.length:]
            return frames[indices]
        else:
            raise NotImplementedError
