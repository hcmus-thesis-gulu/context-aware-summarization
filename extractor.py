import argparse
import os
import time
import numpy as np
import cv2 as cv
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from model.embedder import DINOEmbedder
from model.utils import count_frames


def extract_embedding_from_video(video_path, filename, output_folder,
                                 embedder, frame_rate=None):
    # Define transformations
    transform = ToTensor()
    
    # Get file path
    print(f'Extracting features for {filename}')
    video_name = os.path.splitext(filename)[0]
    video_file = os.path.join(video_path, filename)
    embedding_file = os.path.join(output_folder, f'{video_name}.npy')
    sample_file = os.path.join(output_folder, f'{video_name}_samples.npy')
    if os.path.exists(embedding_file) and os.path.exists(sample_file):
        return
    
    # Extract features for each frame of the video
    cap = cv.VideoCapture(video_file)
    # Get the video's frame rate, total frames
    fps, total_frames = count_frames(video_file)
    
    if frame_rate == None:
        frame_rate = fps
    
    # Calculate the total number of samples
    frame_step = fps // frame_rate
    total_samples = (total_frames + frame_step - 1) // frame_step
    
    # Create holders
    embeddings = np.zeros((total_samples, embedder.emb_dim))
    samples = np.zeros((total_samples), dtype=np.int64)

    pbar = tqdm(total=total_samples)
    
    frame_idx = 0
    result_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step:
            frame_idx += 1
            continue
        
        # Convert frame to PyTorch tensor and extract features
        img = Image.fromarray(frame, mode="RGB")
        img = transform(img).unsqueeze(0)
        embedding = embedder.image_embedding(img)
        
        embeddings[result_idx] = embedding
        samples[result_idx] = frame_idx
        
        result_idx += 1
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save feature embeddings to file
    np.save(embedding_file, embeddings)
    np.save(sample_file, samples)
    

def extract_embedding_from_path(video_path, output_folder,
                                frame_rate=None, representation='cls',
                                model_name='b16', device='cuda'):
    embedder = DINOEmbedder(representation,
                            model_name,
                            device
                            )
    
    # Extract features for each video file
    for filename in os.listdir(video_path):
        if filename.endswith('.mp4'):
            extract_embedding_from_video(video_path, filename, output_folder,
                                         embedder, frame_rate)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Extract DINO features from videos')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--embedding-folder', type=str, required=True,
                        help='Path to folder to store embeddings')
    
    parser.add_argument('--representation', type=str, default='cls',
                        choices=['cls', 'mean'],
                        help='visual type')
    parser.add_argument('--frame-rate', type=int, 
                        help='Number of frames per second to sample from videos')
    
    parser.add_argument('--model-name', type=str, default='b16',
                        choices=['b16', 'b8', 's16', 's8'],
                        help='Name of the DINO model')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run the model on')

    args = parser.parse_args()

    extract_embedding_from_path(args.video_folder, args.embedding_folder,
                                args.frame_rate, args.representation,
                                args.model_name, args.device
                                )
    print("--- %s seconds ---" % (time.time() - start_time))
