'use client';

import { useEffect, useRef, useState } from 'react';

interface FaceResult {
  bbox: [number, number, number, number];
  confidence: number;
  name: string;
  recognition_confidence: number;
  class_id?: number;
}

interface DetectionResponse {
  faces: FaceResult[];
  processing_time: number;
  threshold_used: number;
  pipeline: string;
}

export default function FaceRecognitionPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [faces, setFaces] = useState<FaceResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [apiStatus, setApiStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [frameInterval, setFrameInterval] = useState<number>(1000);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [pipelineInfo, setPipelineInfo] = useState<string>('');

  const fpsCounterRef = useRef<number[]>([]);

  // Start camera
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: 640, 
            height: 480,
            facingMode: 'user'
          } 
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error('Cannot access camera', err);
      }
    };
    startCamera();
  }, []);

  // Draw video to canvas continuously
  const drawLoop = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas && video.readyState === video.HAVE_ENOUGH_DATA) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
    }
    requestAnimationFrame(drawLoop);
  };

  // Draw face overlays
  const drawFaceOverlays = (detectedFaces: FaceResult[]) => {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return;

    const ctx = overlayCanvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    detectedFaces.forEach((face, idx) => {
      const [x, y, width, height] = face.bbox;

      // Color based on recognition
      const isRecognized = face.name !== 'Unknown';
      const boxColor = isRecognized ? '#10b981' : '#ef4444';
      
      // Draw bounding box
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      const label = `${face.name} (${Math.round(face.recognition_confidence * 100)}%)`;
      ctx.font = 'bold 16px system-ui';
      const textMetrics = ctx.measureText(label);
      const padding = 8;
      
      ctx.fillStyle = boxColor;
      ctx.fillRect(x, y - 30, textMetrics.width + padding * 2, 26);

      // Draw text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + padding, y - 9);

      // Draw class ID if available
      if (face.class_id !== undefined && face.class_id !== -1) {
        ctx.font = '12px system-ui';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillText(`ID: ${face.class_id}`, x + padding, y + height + 18);
      }

      // Draw confidence bar
      const barHeight = 4;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.fillRect(x, y + height + 4, width, barHeight);
      
      ctx.fillStyle = boxColor;
      ctx.fillRect(x, y + height + 4, width * face.recognition_confidence, barHeight);
    });
  };

  // Capture and analyze frame
  const captureAndAnalyze = async () => {
    if (!canvasRef.current || isProcessing) return;

    setIsProcessing(true);
    const frameStartTime = performance.now();
    
    try {
      const canvas = canvasRef.current;
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      const response = await fetch('http://localhost:8111/detect-and-recognize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          image: imageData,
          threshold: threshold 
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: DetectionResponse = await response.json();
      
      setFaces(result.faces);
      setProcessingTime(result.processing_time);
      setPipelineInfo(result.pipeline);
      setApiStatus('connected');
      
      drawFaceOverlays(result.faces);
      
      // Calculate FPS
      const frameTime = performance.now() - frameStartTime;
      fpsCounterRef.current.push(1000 / frameTime);
      if (fpsCounterRef.current.length > 10) fpsCounterRef.current.shift();
      const avgFps = fpsCounterRef.current.reduce((a, b) => a + b, 0) / fpsCounterRef.current.length;
      setFps(Math.round(avgFps * 10) / 10);
      
    } catch (error) {
      console.error('Error processing frame:', error);
      setApiStatus('error');
    } finally {
      setIsProcessing(false);
    }
  };

  // Test API connection
  useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await fetch('http://localhost:8111/health');
        if (response.ok) {
          const data = await response.json();
          setPipelineInfo(data.pipeline);
          setApiStatus('connected');
        } else {
          setApiStatus('error');
        }
      } catch (error) {
        setApiStatus('error');
      }
    };

    testConnection();
    const interval = setInterval(testConnection, 5000);
    return () => clearInterval(interval);
  }, []);

  // Setup video canvas
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.onloadeddata = () => {
        if (canvasRef.current && overlayCanvasRef.current) {
          const width = videoRef.current!.videoWidth;
          const height = videoRef.current!.videoHeight;
          
          canvasRef.current.width = width;
          canvasRef.current.height = height;
          overlayCanvasRef.current.width = width;
          overlayCanvasRef.current.height = height;
        }
        drawLoop();
      };
    }
  }, []);

  // Periodic face detection
  useEffect(() => {
    const interval = setInterval(() => {
      if (apiStatus === 'connected' && !isProcessing) {
        captureAndAnalyze();
      }
    }, frameInterval);

    return () => clearInterval(interval);
  }, [apiStatus, isProcessing, frameInterval, threshold]);

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'connected': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      case 'connecting': return 'bg-yellow-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            Face Recognition System
          </h1>
          <p className="text-gray-400">Face Detection + FaceNet Recognition</p>
        </div>

        {/* Status Bar */}
        <div className="bg-gray-800 rounded-lg shadow-xl p-4 mb-6 border border-gray-700">
          <div className="flex flex-wrap gap-4 items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
              <span className="text-gray-300">
                API: <strong className="text-white">
                  {apiStatus === 'connected' ? 'Connected' : apiStatus === 'error' ? 'Error' : 'Connecting...'}
                </strong>
              </span>
            </div>
            
            <div className="text-gray-300">
              Faces: <strong className="text-white">{faces.length}</strong>
            </div>
            
            <div className="text-gray-300">
              FPS: <strong className="text-white">{fps}</strong>
            </div>
            
            {processingTime > 0 && (
              <div className="text-gray-300">
                Backend: <strong className="text-white">{Math.round(processingTime * 1000)}ms</strong>
              </div>
            )}
            
            <div className="text-gray-300">
              <strong className={isProcessing ? 'text-yellow-400' : 'text-green-400'}>
                {isProcessing ? 'Processing...' : 'Ready'}
              </strong>
            </div>
          </div>
          
          {pipelineInfo && (
            <div className="mt-2 text-xs text-gray-400 border-t border-gray-700 pt-2">
              Pipeline: {pipelineInfo}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera View */}
          <div className="lg:col-span-2">
            <div className="relative bg-black rounded-xl overflow-hidden shadow-2xl" style={{ aspectRatio: '4/3' }}>
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover"
              />
              
              <canvas
                ref={canvasRef}
                className="hidden"
              />
              
              <canvas
                ref={overlayCanvasRef}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />
            </div>
          </div>

          {/* Controls Panel */}
          <div className="space-y-4">
            {/* Threshold Control */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-4 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Recognition Threshold: {threshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Less strict (0.1)</span>
                <span>More strict (0.9)</span>
              </div>
            </div>

            {/* Frame Interval Control */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-4 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Update Interval: {frameInterval}ms
              </label>
              <input
                type="range"
                min="200"
                max="2000"
                step="100"
                value={frameInterval}
                onChange={(e) => setFrameInterval(Number(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Fast</span>
                <span>Slow</span>
              </div>
            </div>

            {/* Detection Results */}
            {faces.length > 0 && (
              <div className="bg-gray-800 rounded-lg shadow-xl p-4 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-3">Detected Faces</h3>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {faces.map((face, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-lg border ${
                        face.name === 'Unknown' 
                          ? 'bg-red-900/20 border-red-500/30' 
                          : 'bg-green-900/20 border-green-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-white">{face.name}</span>
                        {face.class_id !== undefined && face.class_id !== -1 && (
                          <span className="text-xs text-gray-400">ID: {face.class_id}</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-400 space-y-1">
                        <div className="flex justify-between">
                          <span>Recognition:</span>
                          <strong className="text-white">{Math.round(face.recognition_confidence * 100)}%</strong>
                        </div>
                        <div className="flex justify-between">
                          <span>Detection:</span>
                          <strong className="text-white">{Math.round(face.confidence * 100)}%</strong>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-6 bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
          <h4 className="font-semibold text-blue-300 mb-2">System Information</h4>
          <ul className="text-sm text-blue-200 space-y-1">
            <li>• <strong>Step 1:</strong> Detects and crops faces (same settings as training)</li>
            <li>• <strong>Step 2:</strong> FaceNet recognizes cropped faces using trained model</li>
            <li>• <strong>Threshold:</strong> Adjust recognition confidence threshold (lower = more lenient)</li>
            <li>• <strong>Interval:</strong> Control update frequency (lower = faster, higher CPU)</li>
            <li>• <strong>Pipeline:</strong> Exact match with prepare_dataset.py → facenet_inference.py</li>
          </ul>
        </div>
      </div>
    </div>
  );
}