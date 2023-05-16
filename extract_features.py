import argparse
import os
import torch
import numpy as np
from torchvision.transforms import ToTensor
from transformers import ViTFeatureExtractor, ViTModel
import cv2
from tqdm import tqdm
import time
from PIL import Image

# Load DINO model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

vit_dim = 768

def extract_embedding_from_image(image):
    # Extract features
    with torch.device(device):
        with torch.no_grad():
            inputs = feature_extractor(images=image, return_tensors="pt")
            if device == 'cuda':
                inputs.to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        return embeddings   

def extract_embedding_from_video(video_path, filename, output_folder, frame_rate=None, representation='cls'):
    # Define transformations
    transform = ToTensor()
    
    # Get file path
    print(f'Extracting features for {filename}')
    video_name = os.path.splitext(filename)[0]
    video_file = os.path.join(video_path, filename)
    feature_file = os.path.join(output_folder, f'{video_name}.npy')
    sample_file = os.path.join(output_folder, f'{video_name}_samples.npy')
    if os.path.exists(feature_file) and os.path.exists(sample_file):
        return
    
    # Extract features for each frame of the video
    cap = cv2.VideoCapture(video_file)
    # Get the video's frame rate, total frames, width, height, channel
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_rate == None:
        frame_rate = fps
    
    # Calculate the total number of samples
    frame_step = fps // frame_rate
    total_samples = (total_frames + frame_step - 1) // frame_step
    
    # Create holders
    frames = np.zeros((total_samples, vit_dim))
    samples = np.zeros((total_samples), dtype=np.int64)

    pbar = tqdm(total=total_samples)
    
    frame_index = 0
    result_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_step:
            frame_index += 1
            continue
        
        # Convert frame to PyTorch tensor and extract features
        img = Image.fromarray(frame, mode="RGB")
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = extract_embedding_from_image(img)
            # features = torch.randn(1, 197, 768)
            # L2 normalize features
            features = features / features.norm(dim=-1, keepdim=True)
            # Apply Softmax with Torch
            features = torch.nn.functional.softmax(features, dim=-1)
            
            if device == 'cuda':
                features = features.detach().cpu().squeeze(0)
            else:
                features = features.squeeze(0)
            
            if representation == 'cls':
                features = features[0]
            else:
                features = torch.mean(features, dim=0)
            
            frames[result_index] = features
            samples[result_index] = frame_index
        
        result_index += 1
        frame_index += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()    
    # Save feature embeddings to file
    np.save(feature_file, frames)
    np.save(sample_file, samples)
    
def extract_features(video_path, output_folder, frame_rate=None, representation='cls'):
    # Extract features for each video file
    for filename in os.listdir(video_path):
        with torch.device(device):
            if filename.endswith('.mp4'):
                extract_embedding_from_video(video_path, filename, output_folder, frame_rate, representation)
                break

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Extract DINO features from videos')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--feature-folder', type=str, required=True,
                        help='Path to folder to store feature embeddings')
    parser.add_argument('--representation', type=str, default='cls',
                        choices=['cls', 'mean'],
                        help='visual type')
    parser.add_argument('--frame-rate', type=int, 
                        help='Number of frames per second to sample from videos')

    args = parser.parse_args()

    extract_features(args.video_folder, args.feature_folder, 
                     args.frame_rate, args.representation)
    print("--- %s seconds ---" % (time.time() - start_time))
