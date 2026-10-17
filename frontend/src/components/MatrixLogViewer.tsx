import React, { useState } from 'react';
import { Copy, Check, ChevronDown, ChevronUp, Cpu, Hash } from 'lucide-react';
import type { CountBoxesApiResponse } from '../types';

interface MatrixLogViewerProps {
  matrixDetails: CountBoxesApiResponse['matrixDetails'] | null;
}

export const MatrixLogViewer: React.FC<MatrixLogViewerProps> = ({ matrixDetails }) => {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);

  if (!matrixDetails) return null;

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(matrixDetails, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const { affineMatrix, inverseMatrix, determinant, cropCoordinates, formula, originalImage, inputParams } = matrixDetails;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-5 py-3.5 bg-slate-950 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-400" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
            Backend Matrix Processing & Inversion Log
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleCopyJson}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 transition"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy Matrix JSON'}
          </button>
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-label="Toggle matrix details"
            className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 transition"
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-5 space-y-5 text-xs font-mono">
          {/* Dual 3x3 Matrices Display */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Forward Matrix M */}
            <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-bold text-sky-400">
                  FORWARD AFFINE MATRIX [M]
                </span>
                <span className="text-[10px] text-slate-500">Image → Viewport</span>
              </div>
              <div className="p-2.5 bg-black/60 rounded-lg border border-slate-800/80 font-mono text-slate-300 space-y-1">
                {affineMatrix.map((row, rIdx) => (
                  <div key={rIdx} className="grid grid-cols-3 text-center">
                    {row.map((val, cIdx) => (
                      <span key={cIdx} className={cIdx === 2 ? 'text-amber-300 font-semibold' : 'text-slate-300'}>
                        {typeof val === 'number' ? val.toFixed(3) : val}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
              <div className="mt-2 text-[10px] text-slate-500 flex justify-between">
                <span>Scale: {inputParams.scale}x</span>
                <span>det(M) = {determinant.toFixed(4)}</span>
              </div>
            </div>

            {/* Inverse Matrix M^(-1) */}
            <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-bold text-emerald-400">
                  INVERSE MATRIX [M⁻¹]
                </span>
                <span className="text-[10px] text-slate-500">Viewport → Source Crop</span>
              </div>
              <div className="p-2.5 bg-black/60 rounded-lg border border-slate-800/80 font-mono text-emerald-300 space-y-1">
                {inverseMatrix.map((row, rIdx) => (
                  <div key={rIdx} className="grid grid-cols-3 text-center">
                    {row.map((val, cIdx) => (
                      <span key={cIdx} className={cIdx === 2 ? 'text-emerald-400 font-semibold' : 'text-slate-300'}>
                        {typeof val === 'number' ? val.toFixed(3) : val}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
              <div className="mt-2 text-[10px] text-slate-500 flex justify-between">
                <span>Inversion: M · M⁻¹ = I</span>
                <span className="text-emerald-500">Valid Invertible</span>
              </div>
            </div>
          </div>

          {/* Clamped Crop Coordinate Breakdown */}
          <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Hash className="w-4 h-4 text-amber-400" />
                <span className="text-xs font-bold text-slate-200">
                  Sharp Crop Coordinates & Boundary Clamping
                </span>
              </div>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                cropCoordinates.isClamped
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              }`}>
                {cropCoordinates.isClamped ? 'BOUNDARY CLAMPED' : 'UNCONSTRAINED FIT'}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mb-3">
              <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">left (X)</span>
                <span className="text-sm font-bold text-emerald-400">{cropCoordinates.left}px</span>
                {cropCoordinates.rawLeft !== cropCoordinates.left && (
                  <span className="text-[9px] text-amber-400/80 block">raw: {cropCoordinates.rawLeft}px</span>
                )}
              </div>
              <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">top (Y)</span>
                <span className="text-sm font-bold text-emerald-400">{cropCoordinates.top}px</span>
                {cropCoordinates.rawTop !== cropCoordinates.top && (
                  <span className="text-[9px] text-amber-400/80 block">raw: {cropCoordinates.rawTop}px</span>
                )}
              </div>
              <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">width</span>
                <span className="text-sm font-bold text-sky-400">{cropCoordinates.width}px</span>
                {cropCoordinates.rawWidth !== cropCoordinates.width && (
                  <span className="text-[9px] text-amber-400/80 block">raw: {cropCoordinates.rawWidth}px</span>
                )}
              </div>
              <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                <span className="text-[10px] text-slate-400 block">height</span>
                <span className="text-sm font-bold text-sky-400">{cropCoordinates.height}px</span>
                {cropCoordinates.rawHeight !== cropCoordinates.height && (
                  <span className="text-[9px] text-amber-400/80 block">raw: {cropCoordinates.rawHeight}px</span>
                )}
              </div>
            </div>

            <div className="p-3 bg-black/60 rounded-lg border border-slate-850 space-y-1 text-[11px] text-slate-400">
              <div className="text-slate-500 font-semibold mb-1 text-[10px] uppercase">Evaluation Traces:</div>
              <div className="text-slate-300">↳ {formula.cropX}</div>
              <div className="text-slate-300">↳ {formula.cropY}</div>
              <div className="text-slate-300">↳ {formula.cropWidth}</div>
              <div className="text-slate-300">↳ {formula.cropHeight}</div>
              <div className="text-[10px] text-slate-500 mt-1">
                Original Image: {originalImage.width}×{originalImage.height}px | Destination: {inputParams.viewportW}×{inputParams.viewportH}px (via Sharp)
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
