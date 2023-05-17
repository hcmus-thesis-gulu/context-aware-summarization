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
    

def mean_embeddings(embeddings):
    return np.mean(embeddings, axis=0)


# Compute the cosine similarity between set of features and its mean
def similarity_score(embeddings, mean=None):
    if mean is None:
        mean = mean_embeddings(embeddings)
    
    return np.dot(embeddings, mean) / (np.linalg.norm(embeddings) *
                                       np.linalg.norm(mean)
                                       )
