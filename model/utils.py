import numpy as np


def mean_features(features):
    return np.mean(features, axis=0)


# Compute the cosine similarity between set of features and its mean
def similarity_score(features, mean=None):
    if mean is None:
        mean = mean_features(features)
    
    return np.dot(features, mean) / (np.linalg.norm(features) *
                                     np.linalg.norm(mean)
                                     )
