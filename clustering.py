import os
import time
import argparse
import numpy as np

from model.propogator import Clusterer
from model.selector import Selector


def read_npy(features_path):
    return np.load(features_path)

def calculate_num_clusters(num_frames, min_len, frame_rate=4):    
    max_clusters = min_len*frame_rate
    num_clusters = max_clusters*2/(1 + np.exp((-10**-3) * num_frames)) - max_clusters
    return int(num_clusters)

def cluster_embeddings(embeddings, method, n_clusters,
                       window_size, min_seg_length,
                       distance, embedding_dim):
    clusterer = Clusterer(method, distance, n_clusters, embedding_dim)
    selector = Selector(window_size, min_seg_length)
    labels, reduced_embeddings = clusterer.cluster(embeddings)
    
    return (labels, selector.select(labels, reduced_embeddings),
            clusterer.num_clusters, reduced_embeddings)




def cluster_videos(embedding_folder, clustering_folder, method,
                   min_len, window_size, min_seg_length, distance,
                   embedding_dim):
    for embedding_name in os.listdir(embedding_folder):
        if embedding_name.endswith('.npy') and not embedding_name.endswith('samples.npy'):
            filename = os.path.splitext(embedding_name)[0]
            embedding_file = os.path.join(embedding_folder, embedding_name)
            embeddings = read_npy(embedding_file)
            print(embeddings.shape[0])
            break
            num_clusters = calculate_num_clusters(embeddings.shape[0], min_len)
            
            sample_file = os.path.join(embedding_folder, f'{filename}_samples.npy')
            samples = read_npy(sample_file)
            keyframes_file = filename + '_keyframes.npy'
            scores_file = filename + '_scores.npy'
            labels_file = filename + '_labels.npy'
            reduced_file = filename + '_reduced.npy'
            
            keyframes_path = os.path.join(clustering_folder, keyframes_file)
            scores_path = os.path.join(clustering_folder, scores_file)
            labels_path = os.path.join(clustering_folder, labels_file)
            reduced_path = os.path.join(clustering_folder, reduced_file)
            
            print(f'Clustering frames of {filename}')
            if os.path.exists(keyframes_path) and os.path.exists(scores_path):
                continue
            
            labels, selections, n_clusters, reduced_embs = cluster_embeddings(embeddings,
                                                                              method,
                                                                              num_clusters,
                                                                              window_size,
                                                                              min_seg_length,
                                                                              distance,
                                                                              embedding_dim
                                                                              )
            
            print(f'Number of clusters: {n_clusters}')
            keyframes = samples[selections[0]]
            
            np.save(keyframes_path, keyframes)
            np.save(scores_path, selections[1])
            np.save(labels_path, labels)
            np.save(reduced_path, reduced_embs)


def main():
    parser = argparse.ArgumentParser(description='Cluster the frames of each video into a cluster using sklearn and numpy.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--clustering-folder', type=str, required=True,
                        help='path to output folder for clustering')
    
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans', 'dbscan', 'gaussian', 'agglo'],
                        help='clustering method')
    parser.add_argument('--max-len', type=int, default=10,
                        help='max length')
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['jensenshannon', 'euclidean', 'cosine'],
                        help='distance metric for clustering')
    parser.add_argument('--embedding-dim', type=int, default=3,
                        help='dimension of embeddings')
    
    parser.add_argument('--window-size', type=int, default=10,
                        help='window size for smoothing')
    parser.add_argument('--min-seg-length', type=int, default=10,
                        help='minimum segment length')
    
    
    args = parser.parse_args()

    cluster_videos(embedding_folder=args.embedding_folder,
                   clustering_folder=args.clustering_folder,
                   method=args.method,
                #    num_clusters=args.num_clusters,
                   window_size=args.window_size,
                   min_seg_length=args.min_seg_length,
                   distance=args.distance,
                   embedding_dim=args.embedding_dim
                   )


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
