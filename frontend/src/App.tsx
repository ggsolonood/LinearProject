'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Package,
  Camera,
  Upload,
  ArrowRight,
  Truck,
  Scan,
  AlertCircle,
  Server,
  Settings,
  Activity,
  Layers,
} from 'lucide-react';

import { ViewportInspector } from './components/ViewportInspector';
import { CameraModal } from './components/CameraModal';
import { ResultsDashboard } from './components/ResultsDashboard';
import { MatrixLogViewer } from './components/MatrixLogViewer';
import { generateSamplePalletSvg, dataUrlToFile, loadSampleBoxFile } from './utils/sampleImages';
import type { CountBoxesApiResponse } from './types';

export default function SmartBoxCounterApp() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [naturalWidth, setNaturalWidth] = useState<number>(0);
  const [naturalHeight, setNaturalHeight] = useState<number>(0);

  const [scale, setScale] = useState<number>(1.0);
  const [translateX, setTranslateX] = useState<number>(0);
  const [translateY, setTranslateY] = useState<number>(0);
  const viewportW = 400;
  const viewportH = 300;

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<CountBoxesApiResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);
  const [modelStatus, setModelStatus] = useState<{ configuredModelUrl: string; status: string } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDefaultSample('box1');
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const res = await fetch('/api/model-status');
      if (res.ok) {
        const data = await res.json();
        setModelStatus(data);
      }
    } catch {
      // ignore
    }
  };

  const loadDefaultSample = async (type: 'box1' | 'pallet' | 'crates' | 'shipping' = 'box1') => {
    try {
      if (type === 'box1') {
        const sample = await loadSampleBoxFile();
        setImageFile(sample.file);
        setImageSrc(sample.url);
        setNaturalWidth(sample.width);
        setNaturalHeight(sample.height);

        const fitScale = Math.min(viewportW / sample.width, viewportH / sample.height);
        setScale(fitScale);
        setTranslateX(Math.round((viewportW - sample.width * fitScale) / 2));
        setTranslateY(Math.round((viewportH - sample.height * fitScale) / 2));
        setAnalysisResult(null);
        setErrorMessage(null);
        return;
      }

      const sampleSvg = generateSamplePalletSvg(type);
      const file = await dataUrlToFile(sampleSvg, `warehouse_${type}_sample.svg`);
      setImageFile(file);
      setImageSrc(sampleSvg);
      setNaturalWidth(1200);
      setNaturalHeight(900);

      const initScale = 0.5;
      setScale(initScale);
      setTranslateX(Math.round((viewportW - 1200 * initScale) / 2));
      setTranslateY(Math.round((viewportH - 900 * initScale) / 2));
      setAnalysisResult(null);
      setErrorMessage(null);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImageFile(file);
      setImageSrc(url);
      setNaturalWidth(img.naturalWidth);
      setNaturalHeight(img.naturalHeight);

      const fitScale = Number(Math.max(0.1, Math.min(viewportW / img.naturalWidth, viewportH / img.naturalHeight)).toFixed(2));
      setScale(fitScale);
      setTranslateX(Math.round((viewportW - img.naturalWidth * fitScale) / 2));
      setTranslateY(Math.round((viewportH - img.naturalHeight * fitScale) / 2));

      setAnalysisResult(null);
      setErrorMessage(null);
    };
    img.src = url;
  };

  const handleCameraCapture = (file: File, dataUrl: string, width: number, height: number) => {
    setImageFile(file);
    setImageSrc(dataUrl);
    setNaturalWidth(width);
    setNaturalHeight(height);

    const fitScale = Number(Math.max(0.1, Math.min(viewportW / width, viewportH / height)).toFixed(2));
    setScale(fitScale);
    setTranslateX(Math.round((viewportW - width * fitScale) / 2));
    setTranslateY(Math.round((viewportH - height * fitScale) / 2));
    setAnalysisResult(null);
    setErrorMessage(null);
  };

  const handleResetTransform = () => {
    if (naturalWidth && naturalHeight) {
      const fitScale = Number(Math.max(0.1, Math.min(viewportW / naturalWidth, viewportH / naturalHeight)).toFixed(2));
      setScale(fitScale);
      setTranslateX(Math.round((viewportW - naturalWidth * fitScale) / 2));
      setTranslateY(Math.round((viewportH - naturalHeight * fitScale) / 2));
    } else {
      setScale(1.0);
      setTranslateX(0);
      setTranslateY(0);
    }
  };

  const handleRunBoxCount = async () => {
    if (!imageFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('scale', scale.toString());
      formData.append('translateX', translateX.toString());
      formData.append('translateY', translateY.toString());
      formData.append('viewportW', viewportW.toString());
      formData.append('viewportH', viewportH.toString());

      const res = await fetch('/api/count-boxes', {
        method: 'POST',
        body: formData,
      });

      const contentType = res.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) {
        const text = await res.text();
        throw new Error(
          `Server returned HTTP ${res.status} (${res.statusText || 'Gateway/Proxy Response'}): ${text.slice(0, 150)}`
        );
      }

      const data: CountBoxesApiResponse = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || `Server returned error status ${res.status}`);
      }

      setAnalysisResult(data);

      setTimeout(() => {
        const resultsEl = document.getElementById('results-section');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err: any) {
      console.error('Box count submission error:', err);
      setErrorMessage(err?.message || 'Failed to process matrix crop and box count.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-slate-950">
      {/* Top Header Navigation */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 shadow-lg shadow-emerald-950 flex items-center justify-center">
              <Package className="w-5 h-5 text-slate-950 font-bold" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-bold tracking-tight text-white uppercase">
                  Smart Box Counter
                </h1>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                  AFFINE MATRIX + SHARP
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Warehouse Inspector // 2D Affine Crop Pipeline for External AI Model
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-400">
              <Server className="w-3.5 h-3.5 text-sky-400" />
              <span>Target Model:</span>
              <span className="text-emerald-400 font-semibold truncate max-w-[200px]">
                {modelStatus?.configuredModelUrl || 'http://localhost:8001/predict'}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setIsCameraOpen(true)}
              aria-label="Open Camera Scanner"
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition"
            >
              <Camera className="w-3.5 h-3.5 text-emerald-400" />
              <span>Camera</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-8">
        {/* Banner Section */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/40 border border-slate-800 p-5 sm:p-6 shadow-xl">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="space-y-1.5 max-w-2xl">
              <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-400 font-semibold flex items-center gap-1.5">
                <Scan className="w-3.5 h-3.5" />
                Zero-Loss Full-Resolution Pipeline
              </span>
              <h2 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
                Inspect Cargo & Pallet Box Stacks with Inverse Matrix Cropping
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
                Frame stacked cartons in the fixed 400×300 viewport. The frontend computes 0 crops; the original uncompressed image and transformation parameters (<code className="text-emerald-300 font-mono">scale</code>, <code className="text-emerald-300 font-mono">tx</code>, <code className="text-emerald-300 font-mono">ty</code>) are evaluated server-side with <code className="text-emerald-300 font-mono">Sharp</code> and forwarded to your external box detection model endpoint (<code className="text-sky-300 font-mono">{modelStatus?.configuredModelUrl || 'http://localhost:8001/predict'}</code>).
              </p>
            </div>

            <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800 shrink-0 space-y-2">
              <span className="text-[10px] font-mono text-slate-400 uppercase block font-semibold">
                Instant Warehouse Demos:
              </span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => loadDefaultSample('box1')}
                  className="px-2.5 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-[11px] font-mono text-amber-300 border border-amber-500/50 font-semibold transition flex items-center gap-1.5"
                >
                  <span>📦</span> 1 Box (Single)
                </button>
                <button
                  type="button"
                  onClick={() => loadDefaultSample('pallet')}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 border border-slate-700/60 transition"
                >
                  Pallet (8)
                </button>
                <button
                  type="button"
                  onClick={() => loadDefaultSample('crates')}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 border border-slate-700/60 transition"
                >
                  Crates (9)
                </button>
                <button
                  type="button"
                  onClick={() => loadDefaultSample('shipping')}
                  className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-300 border border-slate-700/60 transition"
                >
                  Container (12)
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Viewport and Ingestion Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-5 flex flex-col items-center">
            <ViewportInspector
              imageSrc={imageSrc}
              naturalWidth={naturalWidth}
              naturalHeight={naturalHeight}
              scale={scale}
              translateX={translateX}
              translateY={translateY}
              onTransformChange={(newScale, newTx, newTy) => {
                setScale(newScale);
                setTranslateX(newTx);
                setTranslateY(newTy);
              }}
              onReset={handleResetTransform}
              viewportW={viewportW}
              viewportH={viewportH}
            />
          </div>

          <div className="lg:col-span-7 space-y-5">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Truck className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                    Image Source Ingestion
                  </h3>
                </div>
                {naturalWidth > 0 && (
                  <span className="text-[11px] font-mono text-slate-400">
                    Source: <span className="text-emerald-400 font-bold">{naturalWidth}×{naturalHeight}px</span>
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 hover:bg-slate-800/80 hover:border-slate-700 transition flex flex-col items-center justify-center text-center group"
                >
                  <Upload className="w-6 h-6 text-sky-400 mb-2 group-hover:scale-110 transition" />
                  <span className="text-sm font-semibold text-slate-200">Upload Image File</span>
                  <span className="text-[11px] text-slate-400">Select JPG, PNG, WEBP</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  aria-label="Upload warehouse box image"
                />

                <button
                  type="button"
                  onClick={() => setIsCameraOpen(true)}
                  className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 hover:bg-slate-800/80 hover:border-slate-700 transition flex flex-col items-center justify-center text-center group"
                >
                  <Camera className="w-6 h-6 text-emerald-400 mb-2 group-hover:scale-110 transition" />
                  <span className="text-sm font-semibold text-slate-200">Live WebRTC Camera</span>
                  <span className="text-[11px] text-slate-400">Scan via camera sensor</span>
                </button>
              </div>

              {/* External Model Pipeline Notice */}
              <div className="p-3 bg-slate-950/90 rounded-xl border border-slate-800/90 text-[11px] font-mono text-slate-400 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>
                    Backend Pipeline: <span className="text-slate-300">Affine Invert → Sharp 400x300 Extract → Model API</span>
                  </span>
                </div>
                <span className="text-emerald-400 font-semibold text-[10px] bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  READY
                </span>
              </div>

              {/* Displacement Parameters */}
              <div className="bg-slate-950/90 rounded-xl p-3.5 border border-slate-850 font-mono text-xs space-y-2">
                <div className="flex items-center justify-between text-slate-400 text-[11px]">
                  <span>ACTIVE AFFINE DISPLACEMENT PARAMETERS</span>
                  <span className="text-emerald-400">STATUS: LIVE</span>
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 text-center">
                  <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">scale</span>
                    <span className="font-bold text-emerald-400">{scale.toFixed(2)}</span>
                  </div>
                  <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">translateX</span>
                    <span className="font-bold text-sky-400">{translateX}px</span>
                  </div>
                  <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">translateY</span>
                    <span className="font-bold text-sky-400">{translateY}px</span>
                  </div>
                  <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">viewportW</span>
                    <span className="font-bold text-slate-300">{viewportW}px</span>
                  </div>
                  <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
                    <span className="text-[9px] text-slate-400 block">viewportH</span>
                    <span className="font-bold text-slate-300">{viewportH}px</span>
                  </div>
                </div>
              </div>

              {errorMessage && (
                <div className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-500/40 text-xs text-rose-300 flex items-start gap-2.5">
                  <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                  <span>{errorMessage}</span>
                </div>
              )}

              <button
                type="button"
                onClick={handleRunBoxCount}
                disabled={!imageFile || isAnalyzing}
                aria-label="Process crop with matrix and count boxes"
                className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-slate-950 font-extrabold text-sm tracking-wide shadow-xl shadow-emerald-950 disabled:opacity-40 transition-all flex items-center justify-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                    <span>Cropping with Matrix & Analyzing with AI Model...</span>
                  </>
                ) : (
                  <>
                    <Package className="w-4 h-4" />
                    <span>Run 2D Affine Matrix Crop & Count Boxes</span>
                    <ArrowRight className="w-4 h-4 ml-1" />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Results Section */}
        <section id="results-section" className="space-y-6 pt-4">
          <ResultsDashboard
            data={analysisResult}
            isLoading={isAnalyzing}
            onClear={() => setAnalysisResult(null)}
          />

          {analysisResult && (
            <MatrixLogViewer matrixDetails={analysisResult.matrixDetails} />
          )}
        </section>
      </div>

      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCameraCapture}
      />

      <footer className="border-t border-slate-850 bg-slate-950/80 py-6 mt-12 text-center text-xs text-slate-500 font-mono">
        <p>Smart Box Counter // 2D Affine Inversion & Matrix Cropping // External AI Model Forwarding</p>
      </footer>
    </main>
  );
}
