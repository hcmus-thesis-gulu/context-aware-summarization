import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTFeatureExtractor, ViTModel
import cv2
from tqdm import tqdm
import random


# Load DINO model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')


def extract_embedding(img):
    # Extract features
    with torch.no_grad():
        inputs = feature_extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    return embeddings


def extract_features(video_path, output_folder, n_frames_per_second=1):
    # Define transformations
    transform = ToTensor()
    
    # Extract features for each video file
    for filename in os.listdir(video_path):
        if filename.endswith('.mp4'):
            print(f'Extracting features for {filename}')
            video_name = os.path.splitext(filename)[0]
            video_file = os.path.join(video_path, filename)
            output_file = os.path.join(output_folder, f'{video_name}.npy')

            # Extract features for each frame of the video
            cap = cv2.VideoCapture(video_file)
            total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Get the video's frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)

            frames = []
            pbar = tqdm(total=total_frame_count)
            # Calculate the number of frames to skip between samples
            skip_frames = int(fps / n_frames_per_second)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Randomly decide whether to keep this frame or not
                if random.random() < 1.0 / skip_frames:
                    # Do something with the frame here
                    pass

                # Convert frame to PyTorch tensor and extract features
                img = Image.fromarray(frame)
                img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    features = extract_embedding(img)
                    
                    # L2 normalize features
                    features = features / features.norm(dim=-1, keepdim=True)
                    # Apply Softmax with Torch
                    features = torch.nn.functional.softmax(features, dim=-1)
                    
                    frames.append(features.squeeze(0).numpy())
                    
                pbar.update(1)
            
            pbar.close()

            # Save feature embeddings to file
            frames = np.array(frames)
            np.save(output_file, frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINO features from videos')
    parser.add_argument('--video-path', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to folder to store feature embeddings')
    parser.add_argument('--frame-rate', type=str, required=True,
                        help='Number of frames per second to sample from videos')
    args = parser.parse_args()

    extract_features(args.video_path, args.output_folder, args.frame_rate)
