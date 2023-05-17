import torch
from transformers import ViTFeatureExtractor, ViTModel


class DINOEmbedder:
    def __init__(self, feature_type='cls', model_name='b16', device='cpu'):
        # Load DINO model and feature extractor
        self.feature_type = feature_type
        self.model_path = f'facebook/dino-vit{model_name}'
        self.device = device
        self.emb_dim = 768
        
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path)
        self.model = ViTModel.from_pretrained(self.model_path)
        
        self.model.to(self.device)

    def image_embedding(self, image):
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.feature_extractor(images=image,
                                            return_tensors="pt")
            
            outputs = self.model(**inputs.to(self.device))
            features = outputs.last_hidden_state.detach().squeeze(0)
        
        # L2 normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        # Apply Softmax with Torch
        features = torch.nn.functional.softmax(features, dim=-1)
        
        if self.device == 'cuda':
            features = features.cpu()
            
        if self.feature_type == 'cls':
            feature = features[0]
        else:
            feature = torch.mean(features, dim=0)
        
        return feature
