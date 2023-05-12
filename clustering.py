import os
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def read_features(features_path):
    return np.load(features_path)

def l2_normalize_features(features):
    norms = np.linalg.norm(features, axis=1)
    features = features / norms[:, None]
    return features

def cluster_features(features, method, n_clusters):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'gaussian':
        model = GaussianMixture(n_components=n_clusters)
    else:
        raise ValueError('Invalid clustering method')
    features = l2_normalize_features(features)
    model.fit(features)
    labels = model.predict(features)
    
    if method == 'kmeans':
        distances = np.zeros((features.shape[0], n_clusters))
        for i in range(n_clusters):
            centroid = model.cluster_centers_[i]
            for j in range(features.shape[0]):
                distances[j, i] = np.dot(features[j], centroid) / np.linalg.norm(centroid)
        scores = distances
        return labels, scores
    else:
        scores = model.score_samples(features)
        return labels, scores

def main():
    parser = argparse.ArgumentParser(description='Cluster the frames of each video into a cluster using sklearn and numpy.')
    parser.add_argument('feature_folder_path', type=str, help='path to folder containing feature files')
    parser.add_argument('clustering_folder_path', type=str, help='path to output folder for clustering')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'gaussian'], help='clustering method')
    parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters')
    args = parser.parse_args()

    feature_folder_path = args.feature_folder_path
    clustering_folder_path = args.clustering_folder_path
    method = args.method
    n_clusters = args.n_clusters

    for feature_name in os.listdir(feature_folder_path):
        if feature_name.endswith('.npy'):
            feature_path = os.path.join(feature_folder_path, feature_name)
            features = read_features(feature_path)
            labels, scores = cluster_features(features, method, n_clusters)
            output_name_labels = os.path.splitext(feature_name)[0] + '_labels.npy'
            output_name_scores = os.path.splitext(feature_name)[0] + '_scores.npy'
            output_path_labels = os.path.join(clustering_folder_path, output_name_labels)
            output_path_scores = os.path.join(clustering_folder_path, output_name_scores)
            np.save(output_path_labels, labels)
            np.save(output_path_scores, scores)

if __name__ == '__main__':
    main()