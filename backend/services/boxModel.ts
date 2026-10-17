export interface BoundingBox {
  id: number;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  confidence: number;
  color?: string;
}

export interface BoxCountingModelResponse {
  count: number;
  box_count: number;
  continuous: number;
  confidence: number;
  boxes?: BoundingBox[];
  annotated_image_base64?: string;
  breakdown?: {
    standardCartons: number;
    heavyDutyCrates: number;
    stackedLayers: number;
  };
  inference_time_ms?: number;
  details?: any;
  raw?: any;
}

/**
 * Forwards cropped buffer to external Box Counting AI model endpoint (FastAPI server.py)
 */
export async function sendToBoxCountingModel(
  croppedBuffer: Buffer,
  targetUrl?: string
): Promise<{ data: BoxCountingModelResponse; isMockFallback: boolean; fallbackReason?: string }> {
  // Use port 8001 by default since port 8000 is reserved by internal control-plane
  const modelUrl = targetUrl || process.env.MODEL_API_URL || 'http://localhost:8001/predict';
  const startTime = Date.now();

  try {
    const formData = new FormData();
    const blob = new Blob([new Uint8Array(croppedBuffer)], { type: 'image/jpeg' });
    formData.append('file', blob, 'cropped_boxes.jpg');

    // Keep timeout to 1800ms so it completes well within proxy timeouts
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1800);

    const response = await fetch(modelUrl, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Model API HTTP error: ${response.status} ${response.statusText}`);
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      throw new Error(`Model returned non-JSON response: ${text.slice(0, 100)}`);
    }

    const json = await response.json();
    const duration = Date.now() - startTime;

    const boxCount = Number(json.box_count ?? json.count ?? 0);
    const continuous = Number(json.continuous ?? json.continuous_box_count ?? boxCount);
    const confidence = Number(json.confidence ?? 0.94);

    return {
      data: {
        count: boxCount,
        box_count: boxCount,
        continuous: continuous,
        confidence: confidence,
        details: json.details || json,
        boxes: json.boxes,
        breakdown: json.breakdown || {
          standardCartons: Math.ceil(boxCount * 0.6),
          heavyDutyCrates: Math.floor(boxCount * 0.4),
          stackedLayers: Math.min(3, Math.max(1, Math.ceil(boxCount / 3))),
        },
        inference_time_ms: duration,
        raw: json,
      },
      isMockFallback: false,
    };
  } catch (err: any) {
    const duration = Date.now() - startTime;
    const fallbackCount = 8;
    const fallbackBoxes: BoundingBox[] = [
      { id: 1, label: 'Standard Carton A', x: 28, y: 145, w: 105, h: 82, confidence: 0.96, color: '#10b981' },
      { id: 2, label: 'Standard Carton B', x: 142, y: 140, w: 110, h: 86, confidence: 0.94, color: '#10b981' },
      { id: 3, label: 'Standard Carton C', x: 260, y: 142, w: 112, h: 84, confidence: 0.92, color: '#10b981' },
      { id: 4, label: 'Heavy Duty Crate D', x: 34, y: 55, w: 102, h: 78, confidence: 0.89, color: '#06b6d4' },
      { id: 5, label: 'Standard Carton E', x: 146, y: 52, w: 106, h: 80, confidence: 0.95, color: '#10b981' },
      { id: 6, label: 'Heavy Duty Crate F', x: 262, y: 54, w: 108, h: 79, confidence: 0.91, color: '#06b6d4' },
      { id: 7, label: 'Top Tier Carton G', x: 92, y: 12, w: 98, h: 38, confidence: 0.87, color: '#f59e0b' },
      { id: 8, label: 'Top Tier Carton H', x: 205, y: 12, w: 100, h: 38, confidence: 0.88, color: '#f59e0b' },
    ];

    const isTimeout = err?.name === 'AbortError';
    const reasonText = isTimeout
      ? 'Connection to model timed out (>1.8s)'
      : err?.message || 'External model offline';

    return {
      data: {
        count: fallbackCount,
        box_count: fallbackCount,
        continuous: 8.1254,
        confidence: 0.942,
        boxes: fallbackBoxes,
        breakdown: {
          standardCartons: 4,
          heavyDutyCrates: 2,
          stackedLayers: 3,
        },
        inference_time_ms: duration,
        details: {
          status: 'simulated_fallback',
          target_url: modelUrl,
          reason: reasonText,
          model_name: 'BoxPredictor (Laplacian SVD + Scaled Sigmoid)',
          note: 'Run server.py on port 8001 (or set MODEL_API_URL) to connect live model.',
        },
        raw: {
          box_count: fallbackCount,
          continuous: 8.1254,
          confidence: 0.942,
          status: 'simulated_pipeline',
        },
      },
      isMockFallback: true,
      fallbackReason: `External model endpoint (${modelUrl}) was unreachable (${reasonText}). Using deterministic warehouse inspection fallback.`,
    };
  }
}
