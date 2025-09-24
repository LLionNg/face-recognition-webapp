# FaceNet Face Recognition Training & Inference

This project provides a complete pipeline for training and using FaceNet for face recognition, saving models in PyTorch (.pt) format.

## Requirements

### Python Dependencies

```bash
# Core requirements
torch>=1.9.0
torchvision>=0.10.0
facenet-pytorch>=2.5.0
Pillow>=8.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
```

### Installation

```bash
# Install PyTorch (visit https://pytorch.org for specific installation)
# For CPU only:
pip install torch torchvision torchaudio

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install facenet-pytorch opencv-python matplotlib scikit-learn Pillow numpy
```

## Dataset Structure

Organize your face images in the following structure:

```
known_people_photos/
├── 65020733/          # Person ID/Name
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 65020766/
│   ├── image1.jpg
│   └── ...
├── 65020788/
│   └── ...
└── ...
```

**Important**: 
- Use the photo organization script first to flatten your Test/Train subdirectories
- Each person should have their own folder with their ID or name
- Images should be in common formats (.jpg, .jpeg, .png, .bmp)

## Usage

### 1. Organize Photos (if needed)

First, run the photo organization script to flatten your directory structure:

```bash
python organize_photos.py --directory known_people_photos
```

### 2. Train the Model

```bash
# Basic training
python facenet_training.py --data_dir known_people_photos --output my_facenet_model.pt

# Advanced options
python facenet_training.py \
    --data_dir known_people_photos \
    --output my_facenet_model.pt \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --epochs 100
```

**Training Parameters:**
- `--data_dir`: Directory containing face images organized by person
- `--output`: Output path for the trained model (.pt file)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 50)

### 3. Use the Trained Model for Inference

#### Single Image Prediction
```bash
python facenet_inference.py --model my_facenet_model.pt --image test_image.jpg
```

#### Real-time Webcam Recognition
```bash
python facenet_inference.py --model my_facenet_model.pt --webcam
```

#### Video Processing
```bash
python facenet_inference.py --model my_facenet_model.pt --video input_video.mp4 --output output_video.mp4
```

**Inference Parameters:**
- `--model`: Path to trained model (.pt file)
- `--image`: Single image for prediction
- `--video`: Video file for processing
- `--webcam`: Use webcam for real-time recognition
- `--output`: Output path for processed video
- `--confidence`: Confidence threshold (default: 0.7)

## Model Architecture

The FaceNet implementation uses:

1. **Backbone**: InceptionResnetV1 pre-trained on VGGFace2
2. **Feature Extractor**: 512-dimensional face embeddings
3. **Classifier Head**: 
   - Linear layer (512 → 256)
   - ReLU activation
   - Dropout (0.5)
   - Final classification layer (256 → num_classes)

## Features

### Training Pipeline
- ✅ Automatic face detection and cropping using MTCNN
- ✅ Data augmentation and preprocessing
- ✅ Train/validation split (80/20)
- ✅ Learning rate scheduling
- ✅ Best model checkpointing
- ✅ Training history visualization
- ✅ Class mapping preservation

### Inference Pipeline
- ✅ Single image prediction
- ✅ Real-time webcam recognition
- ✅ Video processing with output
- ✅ Multiple face detection per image
- ✅ Confidence-based filtering
- ✅ Bounding box visualization

## Output Files

After training, you'll get:

1. **`my_facenet_model.pt`** - Main model file containing:
   - Model weights
   - Class mappings
   - Training metadata
   - Performance metrics

2. **`best_my_facenet_model.pt`** - Best performing model during training

3. **`my_facenet_model_mappings.json`** - Class mappings in JSON format

4. **`training_history.png`** - Training/validation curves

## Performance Tips

### For Better Training:
- Use at least 10-20 images per person
- Ensure good image quality and variety
- Include different angles, lighting conditions
- Balance the dataset (similar number of images per person)

### For Faster Inference:
- Use GPU if available
- Reduce confidence threshold for stricter matching
- Process every nth frame for video processing
- Resize input images if they're very large

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python facenet_training.py --batch_size 8
   ```

2. **No Faces Detected**
   - Check image quality
   - Ensure faces are clearly visible
   - Try different MTCNN thresholds

3. **Low Accuracy**
   - Increase training epochs
   - Add more training data
   - Reduce learning rate
   - Fine-tune hyperparameters

4. **Import Errors**
   ```bash
   # Make sure all dependencies are installed
   pip install --upgrade torch torchvision facenet-pytorch opencv-python
   ```

## Example Usage Workflow

```bash
# 1. Organize your photos
python organize_photos.py --directory known_people_photos

# 2. Train the model
python facenet_training.py --data_dir known_people_photos --epochs 50

# 3. Test on a single image
python facenet_inference.py --model facenet_model.pt --image test.jpg

# 4. Use webcam for real-time recognition
python facenet_inference.py --model facenet_model.pt --webcam
```

## Model Loading in Custom Code

```python
import torch
from facenet_inference import FaceRecognitionInference

# Load trained model
inference = FaceRecognitionInference('my_facenet_model.pt', confidence_threshold=0.8)

# Predict on image
results = inference.predict_single_image('path/to/image.jpg')

for result in results:
    print(f"Person: {result['person']}, Confidence: {result['confidence']:.3f}")
```

## Web Application (Real-time Face Recognition)

### Frontend Setup (Next.js)

The project includes a Next.js web application for real-time face recognition using your webcam.

#### Prerequisites
```bash
# Install Node.js 18+ from https://nodejs.org
node --version  # Should be v18.0.0 or higher
```

#### Installation & Running

```bash
# Navigate to frontend directory
cd nextjs-frontend

# Install dependencies
npm install
# or
yarn install

# Start development server (Recommended)
npm run dev
# or
yarn dev
```

The web app will be available at `http://localhost:3000`

#### Backend API Setup

The frontend requires the Python FastAPI backend running:

```bash
# In a separate terminal, navigate to backend directory
cd python_backend

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

The API will start at `http://localhost:8111`

#### Using the Web Application

1. **Start Backend**: Run `python app.py` in `python_backend/`
2. **Start Frontend**: Run `npm run dev` in `nextjs-frontend/`
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Allow Camera**: Grant camera permissions when prompted
5. **Adjust Settings**:
   - Recognition Threshold: 0.3-0.5 (lower = more lenient)
   - Update Interval: 1000ms recommended for balance

#### Features

- Real-time face detection using MTCNN
- Live face recognition with FaceNet
- Adjustable confidence threshold
- Performance metrics (FPS, processing time)
- Visual bounding boxes with student IDs
- Multiple face detection support

#### Production Build (Working in progress, might not work)

```bash
# Build for production 
npm run build

# Start production server
npm start
```

#### Configuration

Edit API endpoint in `app/page.tsx` if backend runs on different port:
```typescript
const response = await fetch('http://localhost:8111/detect-and-recognize', {
  // ...
});
```

#### Troubleshooting

**Camera not working:**
- Ensure browser has camera permissions
- Try HTTPS or localhost only
- Check if another app is using the camera

**API connection failed:**
- Verify backend is running on port 8111
- Check CORS settings in `app.py`
- Ensure no firewall blocking localhost

**Low recognition accuracy:**
- Adjust threshold (try 0.3-0.4 for more lenient matching)
- Ensure good lighting on face
- Face camera directly
- Check if model was trained with similar conditions

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 6GB+ VRAM
- **Storage**: ~2GB for model files and dependencies

## Performance Expectations

- **Training Time**: 
  - CPU: 2-4 hours (50 epochs, 1000 images)
  - GPU: 20-30 minutes (50 epochs, 1000 images)
- **Inference Speed**:
  - CPU: ~1-2 seconds per image
  - GPU: ~0.1-0.2 seconds per image