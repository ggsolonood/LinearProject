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
  model_type?: string;
  expected_count?: number;
  top_predictions?: Array<{
    class: number;
    label: string;
    probability: number;
  }>;
  class_distribution?: Record<string, number>;
  entropy?: number;
}

/**
 * Forwards cropped buffer to external Box Counting AI model endpoint (FastAPI server.py)
 */
export async function sendToBoxCountingModel(
  croppedBuffer: Buffer,
  targetUrl?: string
): Promise<{ data: BoxCountingModelResponse }> {
  const modelUrl = targetUrl || process.env.MODEL_API_URL || 'http://localhost:8001/predict';
  const startTime = Date.now();

  try {
    const formData = new FormData();
    const blob = new Blob([new Uint8Array(croppedBuffer)], { type: 'image/jpeg' });
    formData.append('file', blob, 'cropped_boxes.jpg');

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
    const continuous = Number(json.continuous ?? json.continuous_box_count ?? json.expected_count ?? boxCount);
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
        model_type: json.model_type,
        expected_count: json.expected_count,
        top_predictions: json.top_predictions,
        class_distribution: json.class_distribution,
        entropy: json.entropy,
      },
    };
  } catch (err: any) {
    const isTimeout = err?.name === 'AbortError';
    const reasonText = isTimeout
      ? 'Connection to model timed out (>1.8s)'
      : err?.message || 'External model offline';
    throw new Error(`Box counting model unavailable at ${modelUrl}: ${reasonText}`);
  }
}
