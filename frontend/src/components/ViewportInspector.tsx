import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Move,
  Maximize2,
  Crosshair,
  Info,
} from 'lucide-react';

interface ViewportInspectorProps {
  imageSrc: string | null;
  naturalWidth: number;
  naturalHeight: number;
  scale: number;
  translateX: number;
  translateY: number;
  onTransformChange: (scale: number, translateX: number, translateY: number) => void;
  onReset: () => void;
  viewportW?: number;
  viewportH?: number;
}

export const ViewportInspector: React.FC<ViewportInspectorProps> = ({
  imageSrc,
  naturalWidth,
  naturalHeight,
  scale,
  translateX,
  translateY,
  onTransformChange,
  onReset,
  viewportW = 400,
  viewportH = 300,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [initialTranslate, setInitialTranslate] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showGrid, setShowGrid] = useState(true);

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageSrc) return;
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setInitialTranslate({ x: translateX, y: translateY });
  };

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      onTransformChange(scale, initialTranslate.x + dx, initialTranslate.y + dy);
    },
    [isDragging, dragStart, initialTranslate, scale, onTransformChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Global mouse event listeners during drag
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Touch pan handlers
  const handleTouchStart = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!imageSrc || e.touches.length !== 1) return;
    const touch = e.touches[0];
    setIsDragging(true);
    setDragStart({ x: touch.clientX, y: touch.clientY });
    setInitialTranslate({ x: translateX, y: translateY });
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!isDragging || e.touches.length !== 1) return;
    const touch = e.touches[0];
    const dx = touch.clientX - dragStart.x;
    const dy = touch.clientY - dragStart.y;
    onTransformChange(scale, initialTranslate.x + dx, initialTranslate.y + dy);
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  // Wheel zoom handler
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!imageSrc) return;
    e.preventDefault();
    const zoomDelta = e.deltaY < 0 ? 0.05 : -0.05;
    const newScale = Math.min(3.0, Math.max(0.1, Number((scale + zoomDelta).toFixed(2))));

    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const factor = newScale / scale;
      const newTx = mouseX - factor * (mouseX - translateX);
      const newTy = mouseY - factor * (mouseY - translateY);
      onTransformChange(newScale, Math.round(newTx), Math.round(newTy));
    } else {
      onTransformChange(newScale, translateX, translateY);
    }
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newScale = parseFloat(e.target.value);
    onTransformChange(newScale, translateX, translateY);
  };

  const handleStepZoom = (delta: number) => {
    const newScale = Math.min(3.0, Math.max(0.1, Number((scale + delta).toFixed(2))));
    onTransformChange(newScale, translateX, translateY);
  };

  const handleFitCenter = () => {
    if (!naturalWidth || !naturalHeight) return;
    const fitScale = Math.min(viewportW / naturalWidth, viewportH / naturalHeight);
    const roundedScale = Number(Math.max(0.1, fitScale).toFixed(2));
    const tx = Math.round((viewportW - naturalWidth * roundedScale) / 2);
    const ty = Math.round((viewportH - naturalHeight * roundedScale) / 2);
    onTransformChange(roundedScale, tx, ty);
  };

  return (
    <div className="flex flex-col items-center select-none w-full max-w-[440px] mx-auto">
      {/* Viewport Frame with mobile scroll containment */}
      <div className="w-full flex justify-center overflow-x-auto pb-1">
        <div className="relative group p-1 bg-slate-900/90 rounded-2xl border-2 border-slate-700 shadow-2xl overflow-hidden shrink-0">
          {/* HUD Header Bar */}
          <div className="flex items-center justify-between px-3 py-1.5 bg-slate-950/80 text-[11px] font-mono text-slate-400 border-b border-slate-800">
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-semibold text-slate-200 uppercase tracking-wider">
                INSPECTION TARGET // {viewportW}x{viewportH}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowGrid(!showGrid)}
              aria-label="Toggle grid overlay"
              className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
            >
              {showGrid ? 'GRID: ON' : 'GRID: OFF'}
            </button>
          </div>

          {/* The 400x300 Fixed Viewport */}
          <div
            ref={containerRef}
            style={{ width: `${viewportW}px`, height: `${viewportH}px` }}
            className="relative overflow-hidden bg-slate-950 cursor-grab active:cursor-grabbing touch-none"
            onMouseDown={handleMouseDown}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onWheel={handleWheel}
            aria-label="Interactive box targeting viewport"
          >
            {imageSrc ? (
              <img
                src={imageSrc}
                alt="Stacked boxes targeting canvas"
                draggable={false}
                width={naturalWidth > 0 ? naturalWidth : undefined}
                height={naturalHeight > 0 ? naturalHeight : undefined}
                className="absolute top-0 left-0 max-w-none pointer-events-none transition-transform duration-75 ease-out select-none"
                style={{
                  transformOrigin: '0 0',
                  transform: `translate3d(${translateX}px, ${translateY}px, 0) scale(${scale})`,
                  width: naturalWidth > 0 ? `${naturalWidth}px` : undefined,
                  height: naturalHeight > 0 ? `${naturalHeight}px` : undefined,
                }}
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 p-6 text-center space-y-2 pointer-events-none">
                <Crosshair className="w-12 h-12 text-slate-600 animate-pulse stroke-1" />
                <p className="text-xs font-mono uppercase tracking-wider text-slate-400">
                  Awaiting Image Feed
                </p>
                <p className="text-[11px] text-slate-500 max-w-[240px]">
                  Capture via WebRTC Camera, upload a warehouse photo, or select a sample stack.
                </p>
              </div>
            )}

            {/* Targeting Grid Overlay */}
            {showGrid && (
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 border border-emerald-500/30">
                  <div className="border-r border-b border-emerald-500/20" />
                  <div className="border-r border-b border-emerald-500/20" />
                  <div className="border-b border-emerald-500/20" />
                  <div className="border-r border-b border-emerald-500/20" />
                  <div className="border-r border-b border-emerald-500/20" />
                  <div className="border-b border-emerald-500/20" />
                  <div className="border-r border-emerald-500/20" />
                  <div className="border-r border-emerald-500/20" />
                  <div />
                </div>

                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-8 h-8 flex items-center justify-center">
                    <div className="w-full h-[1px] bg-emerald-400/80 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                    <div className="absolute h-full w-[1px] bg-emerald-400/80 shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
                    <div className="absolute w-4 h-4 rounded-full border border-emerald-400/70" />
                  </div>
                </div>

                <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-emerald-400/80" />
                <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-emerald-400/80" />
                <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-emerald-400/80" />
                <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-emerald-400/80" />

                <div className="absolute bottom-2 left-3 bg-slate-950/80 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-mono text-emerald-400 border border-emerald-500/30">
                  TX: {translateX}px | TY: {translateY}px | Z: {scale.toFixed(2)}x
                </div>

                {imageSrc && naturalWidth > 0 && (
                  <div className="absolute top-2 right-3 bg-slate-950/80 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-mono text-slate-300 border border-slate-700/50">
                    SRC: {naturalWidth}×{naturalHeight}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Viewport Control Bar */}
          <div className="p-3 bg-slate-950/90 border-t border-slate-800 space-y-2.5">
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-mono text-slate-400 w-12 flex items-center gap-1">
                <ZoomIn className="w-3.5 h-3.5 text-emerald-400" />
                {scale.toFixed(2)}x
              </span>
              <input
                type="range"
                min="0.10"
                max="3.00"
                step="0.05"
                value={scale}
                onChange={handleSliderChange}
                disabled={!imageSrc}
                aria-label="Viewport zoom scale slider"
                className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:opacity-40"
              />
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => handleStepZoom(-0.1)}
                  disabled={!imageSrc || scale <= 0.15}
                  aria-label="Zoom out"
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40 transition"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => handleStepZoom(0.1)}
                  disabled={!imageSrc || scale >= 2.95}
                  aria-label="Zoom in"
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40 transition"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
              <div className="flex items-center gap-1.5 text-slate-400 text-[11px]">
                <Move className="w-3 h-3 text-slate-500" />
                <span>Drag to Pan</span>
                <span className="text-slate-600">|</span>
                <span>Scroll to Zoom</span>
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={handleFitCenter}
                  disabled={!imageSrc}
                  aria-label="Fit and center image in viewport"
                  className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-medium text-slate-300 disabled:opacity-40 transition"
                >
                  <Maximize2 className="w-3 h-3 text-amber-400" />
                  Fit
                </button>
                <button
                  type="button"
                  onClick={onReset}
                  disabled={!imageSrc}
                  aria-label="Reset viewport transform"
                  className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-medium text-slate-300 disabled:opacity-40 transition"
                >
                  <RotateCcw className="w-3 h-3 text-emerald-400" />
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-[11px] text-slate-400 mt-2 px-1">
        <Info className="w-3.5 h-3.5 text-sky-400 shrink-0" />
        <span>
          Framed target coordinates are sent to backend for 2D Affine Inverse Matrix cropping.
        </span>
      </div>
    </div>
  );
};
