import torch
import json
from pathlib import Path

def check_model_info(model_path):
    """Check what's inside the FaceNet model checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"CHECKING MODEL: {model_path}")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("CHECKPOINT CONTENTS:")
    print("-" * 70)
    for key in checkpoint.keys():
        print(f"  • {key}")
    
    print(f"\nMODEL INFORMATION:")
    print("-" * 70)
    
    # Number of classes
    if 'num_classes' in checkpoint:
        print(f"  Number of classes: {checkpoint['num_classes']}")
    
    # Class mapping
    if 'idx_to_class' in checkpoint:
        idx_to_class = checkpoint['idx_to_class']
        print(f"\n  Total classes in mapping: {len(idx_to_class)}")
        print(f"\n  CLASS MAPPING (idx -> name):")
        print("  " + "-" * 66)
        
        # Sort by index for better readability
        sorted_mapping = sorted(idx_to_class.items(), key=lambda x: int(x[0]) if isinstance(x[0], str) else x[0])
        
        for idx, class_name in sorted_mapping:
            idx_str = str(idx)
            print(f"  [{idx_str:>3}] -> {class_name}")
    else:
        print("No idx_to_class mapping found!")
    
    # Class to index (reverse mapping)
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        print(f"\n  REVERSE MAPPING (name -> idx):")
        print("  " + "-" * 66)
        
        for class_name, idx in sorted(class_to_idx.items()):
            print(f"  {class_name:20s} -> [{idx}]")
    
    # Training info
    print(f"\nTRAINING INFORMATION:")
    print("-" * 70)
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    if 'best_val_acc' in checkpoint:
        print(f"  Best Validation Accuracy: {checkpoint['best_val_acc']:.2%}")
    
    if 'train_acc' in checkpoint:
        print(f"  Training Accuracy: {checkpoint['train_acc']:.2%}")
    
    # Model architecture info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n  Model has {len(state_dict)} layers/parameters")
        
        # Check output layer size
        for key in state_dict.keys():
            if 'weight' in key and key.endswith('weight'):
                layer_name = key.replace('.weight', '')
                shape = state_dict[key].shape
                if len(shape) == 2 and shape[0] == checkpoint.get('num_classes', 0):
                    print(f"  Output layer: {layer_name} -> {shape}")
    
    print(f"\n{'='*70}\n")
    
    # Save mapping to JSON for easy reference
    if 'idx_to_class' in checkpoint:
        output_file = Path(model_path).parent / 'class_mapping.json'
        with open(output_file, 'w') as f:
            json.dump(idx_to_class, f, indent=2)
        print(f"Class mapping saved to: {output_file}\n")
    
    return checkpoint

def check_specific_student(model_path, student_id):
    """Check if a specific student ID exists in the model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'idx_to_class' not in checkpoint:
        print("No class mapping found in model!")
        return
    
    idx_to_class = checkpoint['idx_to_class']
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    
    print(f"\n{'='*70}")
    print(f"SEARCHING FOR STUDENT: {student_id}")
    print(f"{'='*70}\n")
    
    if student_id in class_to_idx:
        idx = class_to_idx[student_id]
        print(f"FOUND!")
        print(f"   Student ID: {student_id}")
        print(f"   Class Index: {idx}")
        print(f"   This student should be recognized if confidence > threshold")
    else:
        print(f"NOT FOUND!")
        print(f"   Student ID '{student_id}' is not in the trained model")
        print(f"\n   Available students:")
        for name in sorted(class_to_idx.keys()):
            print(f"     • {name}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    model_path = "models/best_facenet_model.pt"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--student":
            if len(sys.argv) > 2:
                check_specific_student(model_path, sys.argv[2])
            else:
                print("Usage: python check_model_info.py --student <student_id>")
        else:
            model_path = sys.argv[1]
            check_model_info(model_path)
    else:
        check_model_info(model_path)