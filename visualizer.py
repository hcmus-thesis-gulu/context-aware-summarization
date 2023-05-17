import os
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

tsne = TSNE(n_components=2, perplexity=30.0)
pca = PCA(n_components=2)

def broadcast_video(raw_video_path, frame_indexes, output_path, fps=None):
    raw_video = cv.VideoCapture(raw_video_path)
    width = int(raw_video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(raw_video.get(cv.CAP_PROP_FRAME_HEIGHT))
  
    if fps == None:
        fps = int(raw_video.get(cv.CAP_PROP_FPS))
    
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    video = cv.VideoWriter(output_path, fourcc, float(fps), (width, height))
    current_frame = 0
    pbar = tqdm(total=len(frame_indexes))
    while True:
        ret, frame = raw_video.read()
        if not ret:
            break
        if current_frame in frame_indexes:
            video.write(frame)
            pbar.update(1)
        
        current_frame += 1
      
    raw_video.release()
    video.release()
    pbar.close()

def visualize_video(video_folder, feature_folder, clustering_folder, demo_folder, video_name, fps=None):
    sample_file = os.path.join(feature_folder, f'{video_name}_samples.npy')
    keyframe_file = os.path.join(clustering_folder, f'{video_name}_keyframes.npy')
    sample_video_path = os.path.join(demo_folder, f'{video_name}_sample.avi')
    keyframe_video_path = os.path.join(demo_folder, f'{video_name}_keyframes.avi')
    raw_video_path = os.path.join(video_folder, f'{video_name}.mp4')
    
    try:
        samples = np.load(sample_file)
        broadcast_video(raw_video_path=raw_video_path, frame_indexes=samples, 
                        output_path=sample_video_path, fps=fps)
        keyframes = np.load(keyframe_file)
        broadcast_video(raw_video_path=raw_video_path, frame_indexes=keyframes, 
                        output_path=keyframe_video_path, fps=fps)
    except Exception as e:
        print(e)
        print(f'{video_name} not found')
  
def visualize_cluster(video_folder, feature_folder,
                      clustering_folder, video_name,
                      show_image=False):
    sample_file = os.path.join(feature_folder, f'{video_name}_samples.npy')
    feature_file = os.path.join(feature_folder, f'{video_name}.npy')
    keyframe_file = os.path.join(clustering_folder, f'{video_name}_keyframes.npy')  
    video_file = os.path.join(video_folder, f'{video_name}.mp4')
    
    # try:
    sample_indexes = np.load(sample_file)
    keyframe_indexes = np.load(keyframe_file)
    pbar = tqdm(total=len(sample_indexes))
    
    # Fit and transform the data
    features = np.load(feature_file)
    features_pca = pca.fit_transform(features)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Plot the transformed data
    fig, ax = plt.subplots()
    ax.margins(tight=True)
    ax.scatter(features_tsne[:, 0], features_tsne[:, 1],
               c=sample_indexes, cmap='rainbow', alpha=0.6)
    
    if show_image:
        video = cv.VideoCapture(video_file)
        
        frame_index = 0
        feature_index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            if frame_index in sample_indexes:
                props = dict(edgecolor='red', linewidth=1)
                if frame_index not in keyframe_indexes:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    props = None
                
                imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(frame, zoom=0.02),
                features_tsne[feature_index],
                bboxprops=props
                )
                
                ax.add_artist(imagebox)
                feature_index += 1
                pbar.update(1)
            
            frame_index += 1
        pbar.close()
        plt.show()
    # except Exception as e:
    #   print(e)

  
def main():
    parser = argparse.ArgumentParser(description='Visualize result')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--feature-folder', type=str, required=True,
                        help='path to folder containing feature files')
    parser.add_argument('--clustering-folder', type=str, required=True,
                        help='path to output folder for clustering')
    parser.add_argument('--demo-folder', type=str, required=True,
                        help='path to folder saving demo videos')
    parser.add_argument('--visual-type', type=str, default='cluster',
                        choices=['cluster', 'video'],
                        help='visual type')
    parser.add_argument('--video-name', type=str, help='video name')
    parser.add_argument('--fps', type=int, help='video fps')
    parser.add_argument('--show-image', action='store_true',
                        help='show image in cluster')

    args = parser.parse_args()
    
    if args.visual_type == 'cluster':
        visualize_cluster(video_folder=args.video_folder,
                          feature_folder=args.feature_folder,
                          clustering_folder=args.clustering_folder,
                          video_name=args.video_name,
                          show_image=args.show_image)
    else:
        visualize_video(video_folder=args.video_folder,
                        feature_folder=args.feature_folder,
                        clustering_folder=args.clustering_folder,
                        demo_folder=args.demo_folder,
                        video_name=args.video_name,
                        fps=args.fps)


if __name__ == '__main__':
    main()
