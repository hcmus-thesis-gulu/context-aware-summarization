import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTFeatureExtractor, ViTModel
import cv2
from tqdm import tqdm
import time

# Load DINO model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

vit_dim_x, vit_dim_y = 197, 768

def extract_embedding(img):
    # Extract features
    with torch.device(device):
        with torch.no_grad():
            inputs = feature_extractor(images=img, return_tensors="pt")
            if device == 'cuda':
                inputs.to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        return embeddings   

    # if device == 'cuda':
    #     return embeddings.detach().cpu()


def extract_features(video_path, output_folder, n_frames_per_second=None):
    # Define transformations
    transform = ToTensor()
    
    # Extract features for each video file
    for filename in os.listdir(video_path):
        with torch.device(device):
            if filename.endswith('.mp4'):
                print(f'Extracting features for {filename}')
                video_name = os.path.splitext(filename)[0]
                video_file = os.path.join(video_path, filename)
                feature_file = os.path.join(output_folder, f'{video_name}.npy')
                sample_file = os.path.join(output_folder, f'{video_name}_samples.npy')
                if os.path.exists(feature_file) and os.path.exists(sample_file):
                    continue

                # Extract features for each frame of the video
                cap = cv2.VideoCapture(video_file)
                # Get the video's frame rate, width, height, channel
                fps = cap.get(cv2.CAP_PROP_FPS)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                channel = 3

                # Calculate the number of frames to skip between samples
                if n_frames_per_second:
                    skip_frames = int(fps / n_frames_per_second)
                else:
                    skip_frames = 1
                
                total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frame_count = total_frame_count//skip_frames + int(total_frame_count % skip_frames != 0)
                
                frames = np.zeros((total_frame_count, vit_dim_x, vit_dim_y))
                samples = np.zeros((total_frame_count, height, width, channel))
                pbar = tqdm(total=total_frame_count)
                
                frame_index = -1
                result_index = -1
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_index += 1
                
                    # Decide whether to keep this frame or not
                    if frame_index % skip_frames:
                        # Skip the frame
                        continue
                    result_index += 1
                
                    # Convert frame to PyTorch tensor and extract features
                    img = Image.fromarray(frame, mode="RGB")
                    img = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        features = extract_embedding(img)
                        # features = torch.randn(1, 197, 768)
                        # L2 normalize features
                        features = features / features.norm(dim=-1, keepdim=True)
                        # Apply Softmax with Torch
                        features = torch.nn.functional.softmax(features, dim=-1)
                        if device == 'cuda':
                            features.detach().cpu()
                        frames[result_index] = features.squeeze(0).numpy()
                        samples[result_index] = frame

                    pbar.update(1)

                cap.release()

                # Save feature embeddings to file
                frames = np.array(frames)
                np.save(feature_file, frames)
                samples = np.array(samples)
                np.save(sample_file, samples)
                
                pbar.close()
                break
                

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Extract DINO features from videos')
    parser.add_argument('--video-folder', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--feature-folder', type=str, required=True,
                        help='Path to folder to store feature embeddings')
    parser.add_argument('--frame-rate', type=int,
                        help='Number of frames per second to sample from videos')
    args = parser.parse_args()

    extract_features(args.video_folder, args.feature_folder, args.frame_rate)
    print("--- %s seconds ---" % (time.time() - start_time))
