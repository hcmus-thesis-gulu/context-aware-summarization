import os
import time
import argparse
import numpy as np

from model.propogator import Clusterer
from model.selector import Selector
from model.utils import calculate_num_clusters


def localize_context(embeddings, method, n_clusters, window_size,
                     min_seg_length, distance, embedding_dim):
    clusterer = Clusterer(method, distance, n_clusters, embedding_dim)
    selector = Selector(window_size, min_seg_length)
    labels, reduced_embeddings = clusterer.cluster(embeddings)
    
    return (labels, selector.select(labels),
            clusterer.num_clusters, reduced_embeddings)


def localize_videos(embedding_folder, clustering_folder, method,
                    max_len, window_size, min_seg_length,
                    distance, embedding_dim):
    for embedding_name in os.listdir(embedding_folder):
        if embedding_name.endswith('.npy') and not embedding_name.endswith('samples.npy'):
            print(f"Processing the context of video {embedding_name}")
            
            filename = os.path.splitext(embedding_name)[0]
            embedding_file = os.path.join(embedding_folder, embedding_name)
            embeddings = np.load(embedding_file)
            print(f"The extracted context has {embeddings.shape[0]} embeddings")
            
            # sample_file = os.path.join(embedding_folder, f'{filename}_samples.npy')
            # samples = np.load(sample_file)
            scores_file = filename + '_scores.npy'
            labels_file = filename + '_labels.npy'
            reduced_file = filename + '_reduced.npy'
            
            scores_path = os.path.join(clustering_folder, scores_file)
            labels_path = os.path.join(clustering_folder, labels_file)
            reduced_path = os.path.join(clustering_folder, reduced_file)
            
            print(f'Clustering frames of {filename}')
            if os.path.exists(scores_path):
                continue
            
            num_clusters = calculate_num_clusters(embeddings.shape[0], max_len)
            print(f"Initial number of clusters: {num_clusters}")
            labels, parts, n_clusters, reduced_embs = localize_context(embeddings,
                                                                       method,
                                                                       num_clusters,
                                                                       window_size,
                                                                       min_seg_length,
                                                                       distance,
                                                                       embedding_dim
                                                                       )
            
            print(f'Number of clusters: {n_clusters}')
            
            np.save(scores_path, parts)
            np.save(labels_path, labels)
            np.save(reduced_path, reduced_embs)


def main():
    parser = argparse.ArgumentParser(description='Convert Global Context of Videos into Local Semantics.')
    
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--clustering-folder', type=str, required=True,
                        help='path to output folder for clustering')
    
    parser.add_argument('--method', type=str, default='ours',
                        choices=['kmeans', 'dbscan', 'gaussian',
                                 'agglo', 'ours'],
                        help='clustering method')
    parser.add_argument('--num-clusters', type=int, default=0,
                        help='Number of clusters with 0 being automatic detection')
    parser.add_argument('--max-len', type=int, default=60,
                        help='Maximum length of output summarization in seconds')
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

    localize_videos(embedding_folder=args.embedding_folder,
                    clustering_folder=args.clustering_folder,
                    method=args.method,
                    # num_clusters=args.num_clusters,
                    max_len=args.max_len,
                    window_size=args.window_size,
                    min_seg_length=args.min_seg_length,
                    distance=args.distance,
                    embedding_dim=args.embedding_dim
                    )


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
