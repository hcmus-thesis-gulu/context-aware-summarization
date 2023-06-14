import numpy as np
from model.utils import mean_embeddings, similarity_score


#TODO: Better summaries with keyframes at transitions?
class Summarizer:
    def __init__(self, scoring_mode, kf_mode):
        print(f"Summarizer: {scoring_mode} {kf_mode}")
        
        self.scoring_mode = scoring_mode
        
        self.kf_mode = []
        if scoring_mode == 'mean':
            if 'mean' in kf_mode:
                self.kf_mode.append('mean')
        elif 'middle' in kf_mode:
            self.kf_mode.append('middle')
        elif 'ends' in kf_mode:
            self.kf_mode.append('ends')

    # For each segment, compute the mean features and
    # similarity of all features with the mean
    def score_segments(self, embeddings, segments):
        segment_scores = []
        
        for _, start, end in segments:
            # Get the associated features
            segment_features = embeddings[start:end]
            
            # Calculate the scores for frames in the segment
            if self.scoring_mode == "uniform":
                individual_score = len(segment_features)
                score = [individual_score] * len(segment_features)
            else:
                if self.scoring_mode == "mean":
                    representative = mean_embeddings(segment_features)
                else:
                    representative = segment_features[len(segment_features)
                                                      // 2]
                
                score = similarity_score(segment_features,
                                         representative).tolist()
            
            segment_scores.extend(score)
        
        return np.asarray(segment_scores)

    def select_keyframes(self, segments, scores, length):
        keyframe_indices = []
        
        for _, start, end in segments:
            if 'mean' in self.kf_mode:
                segment_scores = scores[start:end]
                keyframe_indices.append(np.argmax(segment_scores) + start)
            
            if 'middle' in self.kf_mode:
                keyframe_indices.append((start + end) // 2)
                
            if 'ends' in self.kf_mode:
                keyframe_indices.append(start)
                keyframe_indices.append(end - 1)

        if length > 0:
            unselected_indicies = np.setdiff1d(np.arange(len(scores)),
                                               keyframe_indices)
            
            unselected_scores = scores[unselected_indicies]
            
            remained_length = length - len(keyframe_indices)
            unselected_keyframes = np.argpartition(unselected_scores,
                                                   -remained_length)[-remained_length:]
            
            keyframe_indices.extend(unselected_indicies[unselected_keyframes])
        
        return np.sort(keyframe_indices)
