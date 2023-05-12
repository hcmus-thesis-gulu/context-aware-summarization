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


def mean_features(features):
    return np.mean(features, axis=0)


# Compute the cosine similarity between set of features and its mean
def similarity_score(features, mean=None):
    if mean is None:
        mean = mean_features(features)
    
    return np.dot(features, mean) / (np.linalg.norm(features) *
                                     np.linalg.norm(mean)
                                     )


# Segment the video into shots based on the smoothed labels
def segment_frames(labels, window_size=5, min_seg_length=4):
    segments = []   # List of (label, start, end) tuples
    start = 0
    current_label = None
    window_size = min(window_size, len(labels))
    
    for i in range(len(labels)):
        # Smooth the labels by taking the majority label in a window
        # whose length is at least window_size
        if i < (window_size // 2):
            left = 0
            right = window_size
        elif i >= len(labels) - (window_size // 2):
            left = len(labels) - window_size
            right = len(labels)
        else:
            left = i - (window_size // 2)
            right = i + (window_size // 2) + 1
        
        window = labels[left:right]
        
        label = np.bincount(window).argmax()
        
        # Partition the video into segments based on the label
        if i == 0:
            current_label = label
        elif i == len(labels) - 1:
            segments.append((current_label, start, i))
        elif label != current_label:
            # Handle short segments
            if len(segments) > 0 and i - start < min_seg_length:
                current_label = label
                
                # If go back to previous segment,
                # the short one is relabeled with the previous label
                if segments[-1][0] == label:
                    segments[-1][2] = i
                    start = i
                
                # If another segment encountered, divide the segment into two,
                # and add the first half to the previous segment
                # while keeping the second one
                else:
                    middle = (start + i) // 2
                    segments[-1][2] = middle
                    start = middle
            
            # Add the segment to the list of segments
            else:
                segments.append((current_label, start, i))
                start = i
    
    # Post process the segments to merge consecutive segments with the same label
    post_segments = []
    current_label = None
    for label, start, end in segments:
        if current_label is None:
            current_label = label
            post_segments.append((current_label, start, end))
        elif label == current_label:
            post_segments[-1] = (current_label, post_segments[-1][1], end)
        else:
            current_label = label
            post_segments.append((current_label, start, end))
    
    return segments


# For each segment, compute the mean features and
# similarity of all features with the mean
def segment_score(features, segments):
    segment_scores = []
    segment_keyframes = []
    
    for _, start, end in segments:
        # Get the associated features
        segment_features = features[start:end]
        
        # Calculate the similarity with mean
        mean = mean_features(segment_features)
        score = similarity_score(segment_features, mean)
        segment_scores.extend(score.tolist())
        
        # Frame with highest score is chosen as the keyframe
        keyframe_idx = np.argmax(score)
        segment_keyframes.append(start + keyframe_idx)
    
    return np.asarray(segment_keyframes), np.asarray(segment_scores)


def cluster_features(features, method, n_clusters, *args, **kwargs):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'gaussian':
        model = GaussianMixture(n_components=n_clusters)
    else:
        raise ValueError('Invalid clustering method')
    features = l2_normalize_features(features)
    model.fit(features)
    labels = model.predict(features)
    
    segments = segment_frames(labels)
    return segment_score(features, segments)


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
            keyframe_idxs, scores = cluster_features(features, method, n_clusters)
            output_name_labels = os.path.splitext(feature_name)[0] + '_labels.npy'
            output_name_scores = os.path.splitext(feature_name)[0] + '_scores.npy'
            output_path_labels = os.path.join(clustering_folder_path, output_name_labels)
            output_path_scores = os.path.join(clustering_folder_path, output_name_scores)
            np.save(output_path_labels, keyframe_idxs)
            np.save(output_path_scores, scores)

if __name__ == '__main__':
    main()