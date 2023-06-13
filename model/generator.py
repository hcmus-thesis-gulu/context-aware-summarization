import numpy as np
from model.utils import mean_embeddings, similarity_score


#TODO: Better summaries with keyframes at transitions?
class Summarizer:
    def __init__(self, representative):
        self.representative = representative

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

    def select_keyframes(self, scores, length):
        keyframe_indices = np.argpartition(scores, -length)[-length:]
        return np.sort(keyframe_indices)
