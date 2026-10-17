import React, { useRef, useState, useEffect } from 'react';
import { Camera, X, RefreshCw, AlertCircle, Check } from 'lucide-react';

interface CameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File, dataUrl: string, width: number, height: number) => void;
}

export const CameraModal: React.FC<CameraModalProps> = ({
  isOpen,
  onClose,
  onCapture,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      stopCamera();
      return;
    }

    startCamera();

    return () => {
      stopCamera();
    };
  }, [isOpen, selectedDeviceId]);

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const startCamera = async () => {
    setIsInitializing(true);
    setError(null);
    stopCamera();

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera access API is not supported in this browser.');
      }

      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
          : { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }

      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter((d) => d.kind === 'videoinput');
      setDevices(videoInputs);
    } catch (err: any) {
      console.error('Camera error:', err);
      setError(
        err.name === 'NotAllowedError'
          ? 'Camera permission was denied. Please allow camera permissions in your browser settings.'
          : err.message || 'Unable to access camera.'
      );
    } finally {
      setIsInitializing(false);
    }
  };

  const handleCaptureSnapshot = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, width, height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_box_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        stopCamera();
        onCapture(file, dataUrl, width, height);
        onClose();
      }
    }, 'image/jpeg', 0.95);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-800 bg-slate-950">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white tracking-wide">
              Live Warehouse Camera Scanner
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close camera modal"
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Camera Viewfinder */}
        <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden min-h-[340px]">
          {error ? (
            <div className="p-6 text-center max-w-md space-y-3">
              <AlertCircle className="w-10 h-10 text-rose-500 mx-auto" />
              <p className="text-sm font-medium text-rose-200">{error}</p>
              <button
                type="button"
                onClick={startCamera}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Retry Camera
              </button>
            </div>
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-contain max-h-[480px]"
              />

              <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                <div className="w-[70%] h-[70%] border-2 border-dashed border-emerald-400/60 rounded-xl flex items-center justify-center">
                  <div className="w-6 h-6 border-t-2 border-l-2 border-emerald-400 absolute top-2 left-2" />
                  <div className="w-6 h-6 border-t-2 border-r-2 border-emerald-400 absolute top-2 right-2" />
                  <div className="w-6 h-6 border-b-2 border-l-2 border-emerald-400 absolute bottom-2 left-2" />
                  <div className="w-6 h-6 border-b-2 border-r-2 border-emerald-400 absolute bottom-2 right-2" />
                  <span className="text-[11px] font-mono text-emerald-300 bg-slate-950/80 px-2 py-0.5 rounded">
                    FRAME CARGO / BOX STACK
                  </span>
                </div>
              </div>

              {isInitializing && (
                <div className="absolute inset-0 bg-slate-950/70 flex items-center justify-center gap-2 text-xs font-mono text-slate-300">
                  <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                  <span>Connecting to Camera Feed...</span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Modal Footer Controls */}
        <div className="px-5 py-4 border-t border-slate-800 bg-slate-950 flex flex-wrap items-center justify-between gap-3">
          {devices.length > 1 ? (
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:ring-1 focus:ring-emerald-500 focus:outline-none"
              aria-label="Select camera device"
            >
              {devices.map((d, index) => (
                <option key={d.deviceId || index} value={d.deviceId}>
                  {d.label || `Camera ${index + 1}`}
                </option>
              ))}
            </select>
          ) : (
            <span className="text-xs text-slate-500 font-mono">
              WebRTC Live Feed (Direct Sensor)
            </span>
          )}

          <div className="flex items-center gap-2 ml-auto">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-300 hover:bg-slate-800 transition"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleCaptureSnapshot}
              disabled={!stream || isInitializing}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-xs font-bold text-white shadow-lg shadow-emerald-950 disabled:opacity-40 transition"
            >
              <Check className="w-4 h-4" />
              Capture Snapshot
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
