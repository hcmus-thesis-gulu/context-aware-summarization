import os
import argparse
import numpy as np
from model.selector import Clusterer, Selector
import time


def read_npy(features_path):
    return np.load(features_path)


def cluster_features(features, method, n_clusters, window_size, min_seg_length):
    clusterer = Clusterer(method, n_clusters)
    selector = Selector(window_size, min_seg_length)
    labels = clusterer.cluster(features)
    
    return selector.select(labels, features)


def main():
    parser = argparse.ArgumentParser(description='Cluster the frames of each video into a cluster using sklearn and numpy.')
    
    parser.add_argument('--feature-folder', type=str, required=True,
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
    
    args = parser.parse_args()

    feature_folder_path = args.feature_folder
    clustering_folder_path = args.clustering_folder
    method = args.method
    num_clusters = args.num_clusters
    window_size = args.window_size
    min_seg_length = args.min_seg_length

    for feature_name in os.listdir(feature_folder_path):
        if feature_name.endswith('.npy') and not feature_name.endswith('samples.npy'):
            filename = os.path.splitext(feature_name)[0]
            feature_file = os.path.join(feature_folder_path, feature_name)
            features = read_npy(feature_file)
            
            sample_file = os.path.join(feature_folder_path, f'{filename}_samples.npy')
            samples = read_npy(sample_file)
            keyframes_file = filename + '_keyframes.npy'
            scores_file = filename + '_scores.npy'
            
            keyframes_path = os.path.join(clustering_folder_path, keyframes_file)
            scores_path = os.path.join(clustering_folder_path, scores_file)
            
            print(f'Clustering frames of {filename}')
            if os.path.exists(keyframes_path) and os.path.exists(scores_path):
                continue
            
            keyframe_idxs, scores = cluster_features(features, method,
                                                     num_clusters, window_size,
                                                     min_seg_length
                                                     )
            keyframes = samples[keyframe_idxs]
            
            np.save(keyframes_path, keyframes)
            np.save(scores_path, scores)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
