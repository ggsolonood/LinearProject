import React, { useState } from 'react';
import {
  Package,
  Layers,
  ShieldCheck,
  Clock,
  Sparkles,
  Download,
  FileCode,
  BarChart3,
  Award,
} from 'lucide-react';
import type { CountBoxesApiResponse } from '../types';

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

  const { croppedImageBase64, modelResult } = data;

  interface PredictionItem {
    class: number;
    label?: string;
    probability: number;
  }

  const topPredictions: PredictionItem[] =
    (modelResult.topPredictions as PredictionItem[]) ||
    (modelResult.rawModelResponse?.top_predictions as PredictionItem[]) ||
    [];

  const classDistribution: Record<string, number> | null =
    (modelResult.classDistribution as Record<string, number>) ||
    (modelResult.rawModelResponse?.class_distribution as Record<string, number>) ||
    null;

  const entropy =
    modelResult.entropy ??
    modelResult.rawModelResponse?.entropy ??
    modelResult.details?.theoretical_values?.entropy;

  const expectedCount =
    modelResult.expectedCount ??
    modelResult.rawModelResponse?.expected_count ??
    modelResult.continuousBoxCount;

  const handleDownloadCrop = () => {
    const link = document.createElement('a');
    link.href = croppedImageBase64;
    link.download = `sharp_crop_${Date.now()}.jpg`;
    link.click();
  };

  return (
    <div className="space-y-6">
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
            Discrete prediction (1–9)
          </div>
        </div>

        {/* Expected Box Count */}
        <div className="bg-gradient-to-br from-amber-950/40 to-slate-900 border border-amber-500/30 p-4 rounded-xl shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase font-semibold text-amber-400">
              Expected Value
            </span>
            <Layers className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-amber-300 font-mono">
              {typeof expectedCount === 'number'
                ? expectedCount.toFixed(3)
                : typeof modelResult.continuousBoxCount === 'number'
                ? modelResult.continuousBoxCount.toFixed(3)
                : modelResult.boxCount.toFixed(3)}
            </span>
            <span className="text-xs text-amber-400 font-medium">
              E[Y] (mean)
            </span>
          </div>
          <div className="mt-1 text-[10px] text-slate-400 font-mono">
            Softmax probabilistic mean
          </div>
        </div>

        {/* Model Confidence */}
        <div className="bg-gradient-to-br from-sky-950/50 to-slate-900 border border-sky-500/30 p-4 rounded-xl shadow-lg flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="font-mono uppercase font-semibold text-sky-400">
              Model Confidence
            </span>
            <ShieldCheck className="w-4 h-4 text-sky-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white font-mono">
              {(modelResult.confidence * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-sky-400 font-medium">
              prob
            </span>
          </div>
          <div className="mt-1 text-[10px] text-slate-400 font-mono">
            Top Softmax probability
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

      {/* Multi-Class Probability Distribution Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800 gap-2">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-emerald-400" />
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              Class Probability Distribution (Softmax 9 Classes)
            </h4>
          </div>
          <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
            <span>Model: <code className="text-emerald-400">session_20261018_082656</code></span>
            {typeof entropy === 'number' && (
              <span className="px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
                Entropy: {entropy.toFixed(4)} bits
              </span>
            )}
          </div>
        </div>

        {/* Top Candidates Pills */}
        {topPredictions.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-mono text-slate-400 flex items-center gap-1">
              <Award className="w-3.5 h-3.5 text-amber-400" /> Top Candidates:
            </span>
            {topPredictions.map((pred) => {
              const isWinner = pred.class === modelResult.boxCount;
              return (
                <span
                  key={pred.class}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium flex items-center gap-1.5 ${
                    isWinner
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow'
                      : 'bg-slate-800/80 text-slate-300 border border-slate-700'
                  }`}
                >
                  <span className="font-bold">
                    {isWinner ? '★ ' : ''}
                    {pred.label || `${pred.class} boxes`}:
                  </span>
                  <span>{(pred.probability * 100).toFixed(2)}%</span>
                </span>
              );
            })}
          </div>
        )}

        {/* Probability Bars (1 to 9 boxes) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
          {Array.from({ length: 9 }, (_, i) => {
            const classNum = i + 1;
            const key = classNum === 1 ? '1_box' : `${classNum}_boxes`;
            const prob =
              classDistribution?.[key] ??
              topPredictions.find((p) => p.class === classNum)?.probability ??
              0;
            const isWinner = classNum === modelResult.boxCount;

            return (
              <div
                key={classNum}
                className={`p-2.5 rounded-xl border transition-all ${
                  isWinner
                    ? 'bg-emerald-950/40 border-emerald-500/60 shadow-[0_0_12px_rgba(16,185,129,0.2)]'
                    : 'bg-slate-950 border-slate-800/80 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                  <span className={`font-semibold ${isWinner ? 'text-emerald-300 font-bold' : 'text-slate-300'}`}>
                    {isWinner && '★ '}{classNum} กล่อง
                  </span>
                  <span className={`font-mono text-[11px] ${isWinner ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>
                    {(prob * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isWinner ? 'bg-gradient-to-r from-emerald-500 to-teal-400' : 'bg-slate-600'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(prob * 100, 1))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Visual Cropped Preview (Clean 400x300 Image) */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="flex items-center justify-between px-5 py-3.5 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
              Backend Cropped Image Preview (400×300)
            </h4>
          </div>

          <div className="flex items-center gap-2">
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

        <div className="p-6 flex items-center justify-center bg-slate-950/50">
          {/* Cropped Image Container with 400x300 exact dimensions */}
          <div className="relative w-[400px] h-[300px] bg-black rounded-xl overflow-hidden border-2 border-emerald-500/40 shadow-2xl shrink-0">
            <img
              src={croppedImageBase64}
              alt="Sharp 400x300 cropped box result"
              className="w-[400px] h-[300px] object-cover"
            />
          </div>
        </div>
      </div>

      {/* Theoretical Model Analysis Details */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3 font-mono text-xs">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <span className="text-emerald-400 font-bold uppercase tracking-wider text-[11px]">
            Theoretical Analysis Details (MLP Classifier Engine)
          </span>
          <span className="text-slate-400 text-[10px]">
            64 SVD + 3 Standardized Edge Stats → 2-Layer MLP → Softmax (9 Classes)
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Model Architecture</span>
            <span className="font-bold text-amber-300">
              MLP (67 → 32 → 9)
            </span>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Predicted Class (Argmax)</span>
            <span className="font-bold text-sky-400">
              Class {modelResult.boxCount} ({modelResult.boxCount} {modelResult.boxCount === 1 ? 'box' : 'boxes'})
            </span>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Expected Value E[Y]</span>
            <span className="font-bold text-emerald-400">
              {typeof expectedCount === 'number' ? expectedCount.toFixed(3) : 'N/A'}
            </span>
          </div>
          <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span className="text-[10px] text-slate-400 block">Shannon Entropy</span>
            <span className="font-bold text-purple-400">
              {typeof entropy === 'number' ? `${entropy.toFixed(4)} bits` : 'N/A'}
            </span>
          </div>
        </div>
      </div>

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
