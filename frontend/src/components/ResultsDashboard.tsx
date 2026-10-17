import React, { useState } from 'react';
import {
  Package,
  Layers,
  ShieldCheck,
  Eye,
  EyeOff,
  Clock,
  Sparkles,
  Download,
  AlertTriangle,
  FileCode,
} from 'lucide-react';
import type { CountBoxesApiResponse, BoundingBox } from '../types';

interface ResultsDashboardProps {
  data: CountBoxesApiResponse | null;
  isLoading: boolean;
  onClear: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  data,
  isLoading,
  onClear,
}) => {
  const [showBoxes, setShowBoxes] = useState(true);
  const [selectedBoxId, setSelectedBoxId] = useState<number | null>(null);
  const [showRawJson, setShowRawJson] = useState(false);

  if (isLoading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center flex flex-col items-center justify-center space-y-4">
        <div className="relative">
          <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center animate-pulse">
            <Package className="w-8 h-8 text-emerald-400" />
          </div>
          <div className="absolute -inset-1 rounded-2xl border-2 border-emerald-500 border-t-transparent animate-spin" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-white">
            Cropping with Matrix & Analyzing with AI Model...
          </h4>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Evaluating 2D Affine Inverse → Sharp 400x300 extract → AI box inference
          </p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { croppedImageBase64, modelResult, isMockFallback, fallbackReason } = data;
  const boxes = modelResult.boxes || [];

  const handleDownloadCrop = () => {
    const link = document.createElement('a');
    link.href = croppedImageBase64;
    link.download = `sharp_crop_${Date.now()}.jpg`;
    link.click();
  };

  return (
    <div className="space-y-6">
      {/* Fallback Banner if external model was offline */}
      {isMockFallback && (
        <div className="bg-amber-950/40 border border-amber-600/30 rounded-xl p-3.5 flex items-start gap-3 text-xs text-amber-200">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold text-amber-300">
              External Model Endpoint Notice:
            </span>
            <p className="text-amber-200/90 leading-relaxed">
              {fallbackReason || 'External model server offline; running deterministic warehouse inference.'}
            </p>
          </div>
        </div>
      )}

      {/* Primary Detection Summary Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Predicted Box Count (Discrete) */}
        <div className="bg-gradient-to-br from-emerald-950/50 to-slate-900 border border-emerald-500/30 p-4 rounded-xl shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase font-semibold text-emerald-400">Predicted Boxes</span>
            <Package className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white font-mono">
              {modelResult.boxCount}
            </span>
            <span className="text-xs text-emerald-400 font-medium">กล่อง (int)</span>
          </div>
          <div className="mt-1 text-[10px] text-slate-400 font-mono">
            Discrete prediction
          </div>
        </div>

        {/* Continuous Box Count */}
        <div className="bg-gradient-to-br from-amber-950/40 to-slate-900 border border-amber-500/30 p-4 rounded-xl shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase font-semibold text-amber-400">Continuous Count</span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-amber-300 font-mono">
              {typeof modelResult.continuousBoxCount === 'number'
                ? modelResult.continuousBoxCount.toFixed(3)
                : modelResult.boxCount.toFixed(3)}
            </span>
            <span className="text-xs text-amber-400 font-medium">y_hat (float)</span>
          </div>
          <div className="mt-1 text-[10px] text-slate-400 font-mono">
            Scaled Sigmoid Output
          </div>
        </div>

        {/* Confidence Card */}
        <div className="bg-gradient-to-br from-sky-950/50 to-slate-900 border border-sky-500/30 p-4 rounded-xl shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase font-semibold text-sky-400">Decision Confidence</span>
            <ShieldCheck className="w-4 h-4 text-sky-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white font-mono">
              {(modelResult.confidence * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-sky-400 font-medium">score</span>
          </div>
          <div className="mt-1 text-[10px] text-slate-400 font-mono">
            Uncertainty ambiguity
          </div>
        </div>

        {/* Inference Latency */}
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase font-semibold text-purple-400">Pipeline Latency</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white font-mono">
              {modelResult.inferenceTimeMs}
            </span>
            <span className="text-xs text-purple-400 font-medium">ms</span>
          </div>
          <div className="mt-1 text-[10px] text-slate-400 font-mono">
            Matrix + Sharp + Inference
          </div>
        </div>
      </div>

      {/* Visual Cropped Preview & Interactive Bounding Box Overlay */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="flex items-center justify-between px-5 py-3.5 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              Backend Cropped Image Preview & Bounding Box Inspection (400×300)
            </h4>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowBoxes(!showBoxes)}
              aria-label="Toggle bounding boxes"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 transition"
            >
              {showBoxes ? <EyeOff className="w-3.5 h-3.5 text-amber-400" /> : <Eye className="w-3.5 h-3.5 text-emerald-400" />}
              {showBoxes ? 'Hide BBoxes' : 'Show BBoxes'}
            </button>
            <button
              type="button"
              onClick={handleDownloadCrop}
              aria-label="Download cropped image"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 transition"
            >
              <Download className="w-3.5 h-3.5" />
              Download JPG
            </button>
          </div>
        </div>

        <div className="p-4 sm:p-6 flex flex-col md:flex-row items-center justify-center gap-6 bg-slate-950/50">
          {/* Cropped Image Container with 400x300 exact dimensions */}
          <div className="w-full max-w-full overflow-x-auto flex justify-center pb-2 md:pb-0">
            <div className="relative w-[400px] h-[300px] bg-black rounded-xl overflow-hidden border-2 border-emerald-500/40 shadow-2xl shrink-0">
              <img
                src={croppedImageBase64}
                alt="Sharp 400x300 cropped box result"
                className="w-[400px] h-[300px] object-cover pointer-events-none"
              />

              {showBoxes && (
                <div className="absolute inset-0 pointer-events-auto">
                  {boxes.map((box) => {
                    const isHovered = selectedBoxId === box.id;
                    return (
                      <div
                        key={box.id}
                        style={{
                          left: `${box.x}px`,
                          top: `${box.y}px`,
                          width: `${box.w}px`,
                          height: `${box.h}px`,
                          borderColor: box.color || '#10b981',
                        }}
                        onMouseEnter={() => setSelectedBoxId(box.id)}
                        onMouseLeave={() => setSelectedBoxId(null)}
                        className={`absolute border-2 rounded transition-all cursor-pointer ${
                          isHovered
                            ? 'bg-emerald-500/30 scale-[1.02] shadow-[0_0_12px_rgba(16,185,129,0.8)] z-20'
                            : 'bg-emerald-500/10 hover:bg-emerald-500/20 z-10'
                        }`}
                      >
                        <span
                          style={{ backgroundColor: box.color || '#10b981' }}
                          className="absolute -top-3 left-0 px-1 py-0.2 rounded text-[9px] font-mono font-bold text-slate-950 shadow"
                        >
                          #{box.id} {(box.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Detected Box Inventory List */}
          <div className="flex-1 w-full max-h-[300px] overflow-y-auto pr-1 space-y-1.5 font-mono text-xs">
            <div className="flex items-center justify-between pb-1 border-b border-slate-800 text-slate-400 text-[11px]">
              <span>DETECTED ITEMS ({boxes.length})</span>
              <span>CONFIDENCE</span>
            </div>
            {boxes.map((box) => (
              <div
                key={box.id}
                onMouseEnter={() => setSelectedBoxId(box.id)}
                onMouseLeave={() => setSelectedBoxId(null)}
                className={`p-2 rounded-lg border flex items-center justify-between transition cursor-pointer ${
                  selectedBoxId === box.id
                    ? 'bg-emerald-950/60 border-emerald-500 text-emerald-300'
                    : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-850'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: box.color || '#10b981' }}
                  />
                  <span className="font-semibold">#{box.id}</span>
                  <span className="text-slate-400">{box.label}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-500">
                    {box.w}×{box.h}px
                  </span>
                  <span className="text-emerald-400 font-bold">
                    {(box.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Theoretical Model Analysis Details (from predict.py Engine) */}
      {modelResult.details && modelResult.details.theoretical_values && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <span className="text-emerald-400 font-bold uppercase tracking-wider text-[11px]">
              Theoretical Analysis Details (predict.py Engine)
            </span>
            <span className="text-slate-400 text-[10px]">
              Laplacian + SVD Subspace + Scaled Sigmoid
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Scaled Sigmoid (k)</span>
              <span className="font-bold text-amber-300">
                k = {modelResult.details.theoretical_values.scaled_sigmoid_k || 10.0}
              </span>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Logit z</span>
              <span className="font-bold text-sky-400">
                {modelResult.details.theoretical_values.logit_z ?? 'N/A'}
              </span>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Edge Density</span>
              <span className="font-bold text-emerald-400">
                {modelResult.details.theoretical_values.edge_density_pct
                  ? `${modelResult.details.theoretical_values.edge_density_pct}%`
                  : 'N/A'}
              </span>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-400 block">Rounding Ambiguity</span>
              <span className="font-bold text-purple-400">
                {modelResult.details.theoretical_values.rounding_ambiguity ?? 'N/A'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Raw Model Response JSON Viewer */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <button
          type="button"
          onClick={() => setShowRawJson(!showRawJson)}
          className="w-full flex items-center justify-between px-5 py-3.5 bg-slate-950 hover:bg-slate-900 text-left transition"
        >
          <div className="flex items-center gap-2">
            <FileCode className="w-4 h-4 text-sky-400" />
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              Raw AI Model JSON Response Payload
            </span>
          </div>
          <span className="text-xs font-mono text-slate-400">
            {showRawJson ? 'Hide JSON [-]' : 'View JSON [+]'}
          </span>
        </button>

        {showRawJson && (
          <div className="p-4 bg-slate-950 border-t border-slate-800">
            <pre className="p-4 bg-black/80 rounded-xl text-emerald-400 font-mono text-xs overflow-x-auto max-h-60 leading-relaxed">
              {JSON.stringify(modelResult.rawModelResponse, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
