import cv2 as cv
import numpy as np


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
    

def mean_features(features):
    return np.mean(features, axis=0)


# Compute the cosine similarity between set of features and its mean
def similarity_score(features, mean=None):
    if mean is None:
        mean = mean_features(features)
    
    return np.dot(features, mean) / (np.linalg.norm(features) *
                                     np.linalg.norm(mean)
                                     )
