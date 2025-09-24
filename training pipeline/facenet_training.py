import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import json
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    """Custom dataset for face recognition training"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split.lower()
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset and create class mappings"""
        classes = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d))])
        
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        target_subdir = 'Train' if self.split == 'train' else 'Test'
        
        for class_name in classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            target_path = os.path.join(class_dir, target_subdir)
            
            if os.path.exists(target_path) and os.path.isdir(target_path):
                for img_name in os.listdir(target_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(target_path, img_name)
                        self.samples.append((img_path, class_idx))
            else:
                for img_name in os.listdir(class_dir):
                    if (img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and 
                        os.path.isfile(os.path.join(class_dir, img_name))):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} {self.split} samples from {len(classes)} classes")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            dummy_image = torch.zeros(3, 160, 160)
            return dummy_image, label

class FaceNetTrainer:
    """FaceNet training class"""
    
    def __init__(self, data_dir, model_save_path='facenet_model.pt', 
                 batch_size=32, learning_rate=0.001, num_epochs=50):
        self.data_dir = data_dir # Will now be 'cropped_photos'
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets from the pre-processed directory
        self.train_dataset = FaceDataset(data_dir, transform=self.transform, split='train')
        self.val_dataset = FaceDataset(data_dir, transform=self.transform, split='test')
        
        # Ensure both datasets have the same class mappings
        if len(self.train_dataset.samples) == 0:
            raise ValueError("No training samples found in Train/ directories!")
        
        if len(self.val_dataset.samples) == 0:
            logger.warning("No validation samples found in Test/ directories! Using 20% of training data for validation.")
            # Fallback: use training dataset and split it
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size]
            )
        
        # Use the class mappings from training dataset
        if hasattr(self.train_dataset, 'class_to_idx'):
            self.class_to_idx = self.train_dataset.class_to_idx
            self.idx_to_class = self.train_dataset.idx_to_class
        else:
            # Handle case when train_dataset is a Subset (from random_split)
            self.class_to_idx = self.train_dataset.dataset.class_to_idx
            self.idx_to_class = self.train_dataset.dataset.idx_to_class
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Initialize model
        self.num_classes = len(self.class_to_idx)
        self.model = self._create_model()
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def _create_model(self):
        """Create FaceNet model with classification head"""
        # Load pre-trained InceptionResnetV1
        facenet = InceptionResnetV1(pretrained='vggface2').eval()
        
        # Freeze the feature extractor initially
        for param in facenet.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        # InceptionResnetV1 outputs 512-dimensional embeddings
        model = nn.Sequential(
            facenet,
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )
        
        return model.to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Wrap the DataLoader with tqdm
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """Full training loop"""
        logger.info("Starting training...")
        logger.info(f"Model: {self.model}")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Number of classes: {self.num_classes}")
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            logger.info("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_{self.model_save_path}")
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        # Save final model
        self.save_model()
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Plot training history
        self.plot_training_history()
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = self.model_save_path
        
        # Save model state dict, class mappings, and training info
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': self.num_classes,
            'model_architecture': 'InceptionResnetV1_with_classifier',
            'input_size': 160,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
        
        # Also save class mappings as JSON for easy access
        mappings_path = path.replace('.pt', '_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump({
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }, f, indent=2)
        logger.info(f"Class mappings saved to {mappings_path}")
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Training history plot saved as 'training_history.png'")

def load_trained_model(model_path):
    """Load a trained FaceNet model for inference"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    num_classes = checkpoint['num_classes']
    facenet = InceptionResnetV1(pretrained=None)
    model = nn.Sequential(
        facenet,
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['class_to_idx'], checkpoint['idx_to_class']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FaceNet for face recognition')
    parser.add_argument('--data_dir', '-d', default='known_people_photos',
                       help='Directory containing face images organized by person')
    parser.add_argument('--output', '-o', default='facenet_model.pt',
                       help='Output path for trained model')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        exit(1)
    
    # Initialize trainer and start training
    trainer = FaceNetTrainer(
        data_dir=args.data_dir,
        model_save_path=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs
    )
    
    trainer.train()