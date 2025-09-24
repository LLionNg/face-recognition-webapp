// app/api/face-detection/route.ts (Alternative to direct Python API calls)
import { NextRequest, NextResponse } from 'next/server';

interface FaceDetectionRequest {
  image: string;
}

export async function POST(request: NextRequest) {
  try {
    const { image }: FaceDetectionRequest = await request.json();
    
    // Forward request to Python backend
    const pythonResponse = await fetch('http://localhost:8000/detect-and-recognize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image }),
    });

    if (!pythonResponse.ok) {
      throw new Error(`Python API error: ${pythonResponse.status}`);
    }

    const result = await pythonResponse.json();
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Face detection error:', error);
    return NextResponse.json(
      { error: 'Face detection failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Health check - forward to Python backend
  try {
    const response = await fetch('http://localhost:8000/health');
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { status: 'error', message: 'Python backend not available' },
      { status: 500 }
    );
  }
}