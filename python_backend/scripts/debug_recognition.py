import requests
import sys
import base64
from pathlib import Path

def test_image(image_path):
    """Test script for face recognition API"""
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    url = "http://localhost:8111/detect-and-recognize"
    
    print(f"Testing: {image_path}")
    print("-" * 60)
    
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "image": f"data:image/jpeg;base64,{image_base64}"
        }
        
        # Send POST request with JSON
        response = requests.post(
            url, 
            json=payload,  # Use json parameter instead of files
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        result = response.json()
        faces = result.get('faces', [])
        
        if not faces:
            print("No faces detected")
            return
        
        print(f"Found {len(faces)} face(s)\n")
        
        for i, face in enumerate(faces, 1):
            name = face.get('name', 'Unknown')
            rec_conf = face.get('recognition_confidence', 0)
            det_conf = face.get('confidence', 0)  # Changed from detection_confidence
            
            status = "✅" if name != "Unknown" else "❌"
            print(f"{status} Face {i}: {name}")
            print(f"   Recognition: {rec_conf*100:.1f}%")
            print(f"   Detection: {det_conf*100:.1f}%")
            
            # Print bounding box
            bbox = face.get('bbox', [])
            if bbox:
                print(f"   BBox: [{', '.join(map(str, bbox))}]")
            print()
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API. Is it running on port 8111?")
    except Exception as e:
        print(f"Error: {e}")

def test_health():
    """Check API health"""
    try:
        response = requests.get("http://localhost:8111/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("API is running")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   YOLO Loaded: {data.get('yolo_loaded', False)}")
            print(f"   FaceNet Loaded: {data.get('facenet_loaded', False)}")
            print(f"   Known Faces: {data.get('known_faces_count', 0)}")
        else:
            print(f"API error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("API not running on port 8111")
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) < 2:
        print("Face Recognition API Debug Tool")
        print("=" * 60)
        print("\nUsage:")
        print("  python debug_recognition.py <image_path>")
        print("  python debug_recognition.py --health")
        print("\nExamples:")
        print("  python debug_recognition.py photo.jpg")
        print("  python debug_recognition.py test_images/face1.png")
        print("  python debug_recognition.py --health")
        sys.exit(1)
    
    if sys.argv[1] == "--health":
        test_health()
    else:
        test_image(sys.argv[1])

if __name__ == "__main__":
    main()