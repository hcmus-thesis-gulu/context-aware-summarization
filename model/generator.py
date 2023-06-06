import numpy as np
from model.utils import mean_embeddings, similarity_score


class Summarizer:
    def __init__(self, representative, method, length):
        self.representative = representative
        self.method = method
        self.length = length

    # For each segment, compute the mean features and
    # similarity of all features with the mean
    def score_segments(self, embeddings, segments):
        segment_scores = []
        
        for _, start, end in segments:
            # Get the associated features
            segment_features = embeddings[start:end]
            
            # Calculate the similarity with representative
            if self.representative == "mean":
                mean = mean_embeddings(segment_features)
            elif self.representative == "middle":
                mean = segment_features[len(segment_features) // 2]
            
            score = similarity_score(segment_features, mean)
            segment_scores.extend(score.tolist())
        
        return np.asarray(segment_scores)

    def select_keyframes(self, frames, scores):
        if self.method == "max":
            indices = np.argpartition(scores, -self.length)[-self.length:]
            return frames[indices]
        else:
            raise NotImplementedError
