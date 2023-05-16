import scipy.io 
import numpy as np

gt_file = 'data/GT/Saving dolphines.mat'
gt_data = scipy.io.loadmat(gt_file)
     
user_score=gt_data.get('user_score')
nFrames=user_score.shape[0]
nbOfUsers=user_score.shape[1]

print(user_score.shape)

# pd_file = 'output/clustering/Saving dolphines_scores.npy'
# pd_scores = np.load(pd_file)

# print(pd_scores.shape)