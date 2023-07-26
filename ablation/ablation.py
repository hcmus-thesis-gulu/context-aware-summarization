import os
import numpy as np
from model.utils import mean_embeddings, similarity_score

context_folder = "ablation/dino-brute-force/contexts/cosine/pca-tsne/embedding-2/intermediate-50"
embedding_folder = "ablation/dino-brute-force/embeddings"
summary_folder = "ablation/summary"

kf_mode = 'mean'
scoring_mode = 'mean'
kf_mode = 'mean'

def score_segments(embeddings, labels):
    print(embeddings.shape)
    segment_scores = []
    sorted_idx = np.argsort(labels) 
    lennn = sorted_idx.shape[0]
    tmp = 0
    segment_features = []
    for i in range(lennn):
        cur_idx = sorted_idx[i]
        prev_idx = sorted_idx[max(0, i - 1)]
        # print(labels[prev_idx], labels[cur_idx])
        if labels[prev_idx] == labels[cur_idx]:
            segment_features.append(embeddings[cur_idx])
        else:
            tmp += len(segment_features)
            representative = mean_embeddings(np.array(segment_features))
            score = similarity_score(segment_features, representative).tolist()
            segment_scores.extend(score)
            segment_features = [embeddings[cur_idx]]
    
    
    tmp += len(segment_features)
    representative = mean_embeddings(np.array(segment_features))
    score = similarity_score(segment_features, representative).tolist()
    segment_scores.extend(score)

    segment_scores = np.array(segment_scores)
    
    print(embeddings.shape, tmp)
    final_scores = []
    for i in range(lennn):
        idx = np.where(sorted_idx == i)[0][0]
        final_scores.append(segment_scores[idx])
    
    print(final_scores)
    return np.asarray(final_scores)

for embedding_name in os.listdir(embedding_folder):
    file_end = "_reduced.npy"
    if embedding_name.endswith(file_end):
        filename = embedding_name[:-len(file_end)]
        
        embedding_name = os.path.join(embedding_folder, embedding_name)
        embeddings = np.load(embedding_name)
        
        samples_file = filename + '_samples.npy'
        samples_path = os.path.join(embedding_folder, samples_file)
        samples = np.load(samples_path)
        
        labels_path = os.path.join(context_folder, filename + '_labels.npy')
        labels = np.load(labels_path)
        
        scores_file = filename + '_scores.npy'
        scores_path = os.path.join(summary_folder, scores_file)
        
        # if os.path.exists(scores_path):
        #     continue
        
        scores = score_segments(embeddings=embeddings, labels=labels)
        # break
            
        sampled_scores = [[sample, score]
                              for sample, score in zip(samples, scores)
                              ]
            
        sorted_scores = np.array(sorted(sampled_scores,
                                              key=lambda x: x[0]))
        print(sorted_scores)
        np.save(scores_path, np.asarray(sorted_scores))

# for filename in os.listdir(labels_folder):
#     if not filename.endswith("_labels.npy"):
#         continue
#     filepath = os.path.join(labels_folder, filename)
#     labels = np.load(filepath)
#     print(labels)
#     sorted_idx = np.argsort(labels)        
#     print(sorted_idx)
#     len = labels.shape[0]
    
#     for i in range(len):
#         cur_idx = sorted_idx[i]
#         prev_idx = max(0, cur_idx - 1)
#         segment_features = []
#         if labels[prev_idx] == labels[cur_idx]:
            
#     break