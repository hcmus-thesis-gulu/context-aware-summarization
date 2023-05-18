import cv2 as cv
import numpy as np
# Probability distribution distance
from sklearn.metrics.pairwise import chi2_kernel, cosine_distances, euclidean_distances
from scipy.spatial.distance import jensenshannon


def count_frames(video_path):
    # Extract features for each frame of the video
    video = cv.VideoCapture(video_path)
    # Get the video's frame rate, total frames
    fps = int(video.get(cv.CAP_PROP_FPS))
    
    count = 0
    while True:
        ret, _ = video.read()
        if not ret:
            break
        count += 1
    
    video.release()
    return fps, count
    

def mean_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


def distance_metric(distance):
    if distance == 'chi2':
        return chi2_kernel
    elif distance == 'jensenshannon':
        return jensenshannon
    elif distance == 'euclidean':
        return euclidean_distances
    elif distance == 'cosine':
        return cosine_distances
    else:
        raise ValueError(f'Unknown distance metric: {distance}')


# Compute the cosine similarity between set of features and its mean
def similarity_score(embeddings, mean=None):
    if mean is None:
        mean = mean_embeddings(embeddings)
    
    return np.dot(embeddings, mean) / (np.linalg.norm(embeddings) *
                                       np.linalg.norm(mean)
                                       )
