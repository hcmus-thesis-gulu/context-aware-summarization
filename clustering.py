import os
import time
import argparse
import numpy as np
from model.selector import Clusterer, Selector


def read_npy(features_path):
    return np.load(features_path)


def cluster_embeddings(features, method, n_clusters,
                       window_size, min_seg_length,
                       distance):
    clusterer = Clusterer(method, distance, n_clusters)
    selector = Selector(window_size, min_seg_length)
    labels = clusterer.cluster(features)
    
    return labels, selector.select(labels, features)


def cluster_videos(embedding_folder, clustering_folder, method,
                   num_clusters, window_size, min_seg_length, distance):
    for embedding_name in os.listdir(embedding_folder):
        if embedding_name.endswith('.npy') and not embedding_name.endswith('samples.npy'):
            filename = os.path.splitext(embedding_name)[0]
            embedding_file = os.path.join(embedding_folder, embedding_name)
            embeddings = read_npy(embedding_file)
            
            sample_file = os.path.join(embedding_folder, f'{filename}_samples.npy')
            samples = read_npy(sample_file)
            keyframes_file = filename + '_keyframes.npy'
            scores_file = filename + '_scores.npy'
            labels_file = filename + '_labels.npy'
            
            keyframes_path = os.path.join(clustering_folder, keyframes_file)
            scores_path = os.path.join(clustering_folder, scores_file)
            labels_path = os.path.join(clustering_folder, labels_file)
            
            print(f'Clustering frames of {filename}')
            if os.path.exists(keyframes_path) and os.path.exists(scores_path):
                continue
            
            labels, selections = cluster_embeddings(embeddings, method,
                                                    num_clusters,
                                                    window_size,
                                                    min_seg_length,
                                                    distance)
            
            keyframes = samples[selections[0]]
            
            np.save(keyframes_path, keyframes)
            np.save(scores_path, selections[1])
            np.save(labels_path, labels)


def main():
    parser = argparse.ArgumentParser(description='Cluster the frames of each video into a cluster using sklearn and numpy.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--clustering-folder', type=str, required=True,
                        help='path to output folder for clustering')
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans', 'dbscan', 'gaussian'],
                        help='clustering method')
    parser.add_argument('--num-clusters', type=int, default=10,
                        help='number of clusters')
    parser.add_argument('--window-size', type=int, default=10,
                        help='window size for smoothing')
    parser.add_argument('--min-seg-length', type=int, default=10,
                        help='minimum segment length')
    parser.add_argument('--distance', type=str, default='euclidean',
                        choices=['jensenshannon', 'euclidean', 'cosine'],
                        help='distance metric for clustering')
    
    args = parser.parse_args()

    cluster_videos(embedding_folder=args.embedding_folder,
                   clustering_folder=args.clustering_folder,
                   method=args.method,
                   num_clusters=args.num_clusters,
                   window_size=args.window_size,
                   min_seg_length=args.min_seg_length,
                   distance=args.distance
                   )


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
