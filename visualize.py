import os
import argparse
import numpy as np
import cv2

def broadcast_video(tensor, output_path):
  nframes, height, width, channel = tensor.shape
  fps = 1
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  video = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

  for i in range(nframes):
    video.write(tensor[i])
    
  video.release()
    

def visualize_video(feature_folder, clustering_folder, demo_folder, video_name):
  sample_file = os.path.join(feature_folder, f'{video_name}_samples.npy')
  keyframe_file = os.path.join(clustering_folder, f'{video_name}_keyframes.npy')
  sample_video_path = os.path.join(demo_folder, f'{video_name}_sample.avi')
  keyframe_video_path = os.path.join(demo_folder, f'{video_name}_keyframes.avi')
  
  try:
    samples = np.load(sample_file)
    broadcast_video(samples, sample_video_path)
    keyframes = np.load(keyframe_file)
    broadcast_video(keyframes, keyframe_video_path)
  except:
    print(f'{video_name} not found')
  
  
def main():
  parser = argparse.ArgumentParser(description='Visualize result')
    
  parser.add_argument('--feature-folder', type=str, required=True,
                        help='path to folder containing feature files')
  parser.add_argument('--clustering-folder', type=str, required=True,
                        help='path to output folder for clustering')
  parser.add_argument('--demo-folder', type=str, required=True,
                        help='path to folder saving demo videos')
  parser.add_argument('--visual-type', type=str, default='cluster', required=True,
                        choices=['cluster', 'video'],
                        help='visual type')
  parser.add_argument('--video-name', type=str,
                        help='video name')

    
  args = parser.parse_args()
  
  if args.visual_type == 'cluster':
    print("he")
  else:
    visualize_video(args.feature_folder, args.clustering_folder, 
                    args.demo_folder, args.video_name)
    

    

if __name__ == '__main__':
    main()