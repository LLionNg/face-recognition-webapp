import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import json
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def load_facenet_model(model_path):
    """Load the trained FaceNet model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = checkpoint['num_classes']
    
    facenet = InceptionResnetV1(pretrained='vggface2')
    model = nn.Sequential(
        facenet,
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded with {num_classes} classes (ignored pretrained logits mismatch)")
    
    return model, checkpoint.get('class_to_idx', {}), checkpoint.get('idx_to_class', {})

def extract_embedding(image_path, model, transform):
    """Extract 512-dim embedding from an image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        embedding_output = []
        
        def hook(module, input, output):
            embedding_output.append(output)
        
        handle = model[0].last_bn.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = model(image_tensor)
        
        handle.remove()
        
        embedding = embedding_output[0].squeeze().cpu().numpy().tolist()
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding from {image_path}: {e}")
        return None

def generate_known_faces_database(model_path, photos_dir, output_file='known_faces.json'):
    """Generate known_faces.json from trained model and photos"""
    
    logger.info("Loading trained FaceNet model...")
    embedding_model, class_to_idx, idx_to_class = load_facenet_model(model_path)
    
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    known_faces = {}
    
    for person_id in os.listdir(photos_dir):
        person_path = os.path.join(photos_dir, person_id)
        
        if not os.path.isdir(person_path):
            continue
        
        logger.info(f"Processing {person_id}...")
        embeddings = []
        
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, img_file)
                embedding = extract_embedding(img_path, embedding_model, transform)
                
                if embedding:
                    embeddings.append(embedding)
        
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            known_faces[person_id] = avg_embedding
            logger.info(f"  Added {person_id} with {len(embeddings)} embeddings")
    
    with open(output_file, 'w') as f:
        json.dump(known_faces, f, indent=2)
    
    logger.info(f"Saved {len(known_faces)} people to {output_file}")
    return known_faces

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained FaceNet model')
    parser.add_argument('--photos', required=True, help='Directory with person photos')
    parser.add_argument('--output', default='known_faces.json', help='Output file')
    
    args = parser.parse_args()
    
    generate_known_faces_database(args.model, args.photos, args.output)