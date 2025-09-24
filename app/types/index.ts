// app/types/index.ts
export interface FaceResult {
  bbox: [number, number, number, number]; // [x, y, width, height]
  confidence: number;
  name: string;
  recognition_confidence: number;
}

export interface DetectionResponse {
  faces: FaceResult[];
  processing_time: number;
}

export interface ApiHealth {
  status: string;
  yolo_loaded: boolean;
  facenet_loaded: boolean;
  known_faces_count: number;
}