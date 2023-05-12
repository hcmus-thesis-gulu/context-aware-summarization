import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import DINOHead, DINOBackbone

def extract_features(video_path, output_folder):
    # Load DINO model
    model = DINOBackbone.from_pretrained('facebook/dino-vitb16')
    head = DINOHead(768, 1024)
    model = torch.nn.Sequential(model, head)

    # Set up transforms
    transform = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Extract features for each video file
    for filename in os.listdir(video_path):
        if filename.endswith('.mp4'):
            print(f'Extracting features for {filename}')
            video_name = os.path.splitext(filename)[0]
            video_file = os.path.join(video_path, filename)
            output_file = os.path.join(output_folder, f'{video_name}.npy')

            # Extract features for each frame of the video
            cap = cv2.VideoCapture(video_file)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to tensor and extract features
                img = Image.fromarray(frame)
                img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    features = model(img).squeeze().numpy()
                    frames.append(features)

            # Save feature embeddings to file
            frames = np.array(frames)
            np.save(output_file, frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINO features from videos')
    parser.add_argument('--video-path', type=str, required=True,
                        help='Path to folder containing videos')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to folder to store feature embeddings')
    args = parser.parse_args()

    extract_features(args.video_path, args.output_folder)