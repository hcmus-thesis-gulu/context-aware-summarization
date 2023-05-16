import scipy.io 
import numpy as np

groundtruth_file = 'data/GT/Saving dolphines.mat'
groundtruth_data = scipy.io.loadmat(groundtruth_file)
     
user_score=groundtruth_data.get('user_score')
nFrames=user_score.shape[0]
nbOfUsers=user_score.shape[1]

print(user_score.shape)

prediction_file = 'output/clustering/Saving dolphines_scores.npy'
prediction_scores = np.load(prediction_file)

print(prediction_scores.shape)