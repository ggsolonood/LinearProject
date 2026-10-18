/**
 * Realistic Warehouse Box Stack Samples (Data URLs) for instant testing
 */

function toBase64Svg(svgString: string): string {
  if (typeof Buffer !== 'undefined') {
    return `data:image/svg+xml;base64,${Buffer.from(svgString).toString('base64')}`;
  }
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgString)))}`;
}

export function generateSamplePalletSvg(type: 'pallet' | 'crates' | 'shipping'): string {
  if (type === 'pallet') {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900">
  <defs>
    <linearGradient id="palletBg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="60%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>
    <linearGradient id="boxGold" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#b45309"/>
    </linearGradient>
    <linearGradient id="boxAmber" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#d97706"/>
      <stop offset="100%" stop-color="#92400e"/>
    </linearGradient>
    <linearGradient id="boxBlue" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </linearGradient>
  </defs>

  <!-- Background Bay -->
  <rect width="1200" height="900" fill="url(#palletBg)"/>
  <rect y="680" width="1200" height="220" fill="#1e293b"/>
  <line x1="0" y1="680" x2="1200" y2="680" stroke="#f59e0b" stroke-width="4" stroke-dasharray="24 16"/>

  <!-- Shelving Uprights -->
  <rect x="120" y="50" width="24" height="630" fill="#3b82f6"/>
  <rect x="1056" y="50" width="24" height="630" fill="#3b82f6"/>
  <line x1="120" y1="130" x2="1080" y2="130" stroke="#1d4ed8" stroke-width="12"/>

  <!-- Pallet Base -->
  <g id="pallet-base">
    <rect x="220" y="665" width="760" height="40" fill="#78350f" rx="3"/>
    <rect x="250" y="680" width="80" height="25" fill="#451a03"/>
    <rect x="420" y="680" width="80" height="25" fill="#451a03"/>
    <rect x="580" y="680" width="80" height="25" fill="#451a03"/>
    <rect x="740" y="680" width="80" height="25" fill="#451a03"/>
    <rect x="880" y="680" width="80" height="25" fill="#451a03"/>
  </g>

  <!-- Layer 1: 3 Bottom Heavy Cartons -->
  <g id="layer-1">
    <rect x="240" y="480" width="230" height="185" fill="url(#boxAmber)" stroke="#78350f" stroke-width="4" rx="6"/>
    <rect x="270" y="515" width="85" height="45" fill="#ffffff" rx="4"/>
    <text x="282" y="545" font-family="monospace" font-weight="900" font-size="16" fill="#0f172a">#B-101</text>
    <line x1="240" y1="580" x2="470" y2="580" stroke="#78350f" stroke-width="12" opacity="0.5"/>

    <rect x="485" y="480" width="230" height="185" fill="url(#boxGold)" stroke="#78350f" stroke-width="4" rx="6"/>
    <rect x="515" y="515" width="85" height="45" fill="#ffffff" rx="4"/>
    <text x="527" y="545" font-family="monospace" font-weight="900" font-size="16" fill="#0f172a">#B-102</text>
    <line x1="485" y1="580" x2="715" y2="580" stroke="#78350f" stroke-width="12" opacity="0.5"/>

    <rect x="730" y="480" width="230" height="185" fill="url(#boxAmber)" stroke="#78350f" stroke-width="4" rx="6"/>
    <rect x="760" y="515" width="85" height="45" fill="#ffffff" rx="4"/>
    <text x="772" y="545" font-family="monospace" font-weight="900" font-size="16" fill="#0f172a">#B-103</text>
    <line x1="730" y1="580" x2="960" y2="580" stroke="#78350f" stroke-width="12" opacity="0.5"/>
  </g>

  <!-- Layer 2: 3 Middle Units -->
  <g id="layer-2">
    <rect x="250" y="305" width="220" height="165" fill="url(#boxBlue)" stroke="#0369a1" stroke-width="4" rx="8"/>
    <rect x="280" y="335" width="85" height="40" fill="#ffffff" rx="3"/>
    <text x="290" y="362" font-family="monospace" font-weight="900" font-size="15" fill="#0c4a6e">#CR-201</text>
    <rect x="265" y="405" width="190" height="20" fill="#075985" rx="3"/>

    <rect x="490" y="305" width="220" height="165" fill="url(#boxGold)" stroke="#78350f" stroke-width="4" rx="6"/>
    <rect x="520" y="335" width="85" height="40" fill="#ffffff" rx="3"/>
    <text x="532" y="362" font-family="monospace" font-weight="900" font-size="15" fill="#0f172a">#B-202</text>
    <line x1="490" y1="390" x2="710" y2="390" stroke="#78350f" stroke-width="10" opacity="0.5"/>

    <rect x="730" y="305" width="220" height="165" fill="url(#boxBlue)" stroke="#0369a1" stroke-width="4" rx="8"/>
    <rect x="760" y="335" width="85" height="40" fill="#ffffff" rx="3"/>
    <text x="770" y="362" font-family="monospace" font-weight="900" font-size="15" fill="#0c4a6e">#CR-203</text>
    <rect x="745" y="405" width="190" height="20" fill="#075985" rx="3"/>
  </g>

  <!-- Layer 3: 2 Top Cartons -->
  <g id="layer-3">
    <rect x="360" y="155" width="230" height="140" fill="url(#boxGold)" stroke="#854d0e" stroke-width="4" rx="6"/>
    <rect x="390" y="185" width="80" height="35" fill="#ffffff" rx="3"/>
    <text x="402" y="210" font-family="monospace" font-weight="900" font-size="14" fill="#0f172a">#B-301</text>
    <line x1="360" y1="225" x2="590" y2="225" stroke="#78350f" stroke-width="8" opacity="0.5"/>

    <rect x="610" y="155" width="230" height="140" fill="url(#boxGold)" stroke="#854d0e" stroke-width="4" rx="6"/>
    <rect x="640" y="185" width="80" height="35" fill="#ffffff" rx="3"/>
    <text x="652" y="210" font-family="monospace" font-weight="900" font-size="14" fill="#0f172a">#B-302</text>
    <line x1="610" y1="225" x2="840" y2="225" stroke="#78350f" stroke-width="8" opacity="0.5"/>
  </g>

  <!-- Signage -->
  <text x="600" y="90" text-anchor="middle" font-family="sans-serif" font-weight="900" font-size="32" fill="#38bdf8" letter-spacing="4">
    BAY 04 // PALLET LOAD #9842 (8 CARTONS)
  </text>
</svg>`;
    return toBase64Svg(svg);
  }

  if (type === 'crates') {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900">
  <rect width="1200" height="900" fill="#0f172a"/>
  <rect y="700" width="1200" height="200" fill="#1e293b"/>
  <g>
    ${[0, 1, 2].map(row => [0, 1, 2].map(col => `
      <g transform="translate(${250 + col * 240}, ${200 + row * 160})">
        <rect width="220" height="145" fill="#1d4ed8" stroke="#60a5fa" stroke-width="4" rx="8"/>
        <rect x="15" y="15" width="190" height="115" fill="#2563eb" rx="4"/>
        <rect x="30" y="30" width="70" height="35" fill="#ffffff" rx="2"/>
        <text x="40" y="55" font-family="monospace" font-size="14" font-weight="bold" fill="#1e3a8a">CR-${row + 1}${col + 1}</text>
      </g>
    `).join('')).join('')}
  </g>
  <text x="600" y="130" text-anchor="middle" font-family="sans-serif" font-weight="800" font-size="28" fill="#60a5fa">MODULAR HEAVY CRATES MATRIX (9 UNITS)</text>
</svg>`;
    return toBase64Svg(svg);
  }

  // shipping container
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900">
  <rect width="1200" height="900" fill="#090d16"/>
  <rect x="140" y="140" width="920" height="640" fill="#1e293b" stroke="#f97316" stroke-width="8" rx="8"/>
  <g transform="translate(170, 170)">
    ${[0, 1, 2, 3].map(c => [0, 1, 2].map(r => `
      <g transform="translate(${c * 215}, ${r * 185})">
        <rect width="200" height="165" fill="#d97706" stroke="#78350f" stroke-width="4" rx="6"/>
        <rect x="25" y="25" width="85" height="45" fill="#ffffff" rx="3"/>
        <text x="35" y="55" font-family="monospace" font-size="13" font-weight="bold" fill="#0f172a">CTN-${c + 1}${r + 1}</text>
      </g>
    `).join('')).join('')}
  </g>
  <text x="600" y="95" text-anchor="middle" font-family="sans-serif" font-weight="800" font-size="28" fill="#fb923c">FREIGHT CARGO CONTAINER #402 (12 CARTONS)</text>
</svg>`;
  return toBase64Svg(svg);
}

export async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || 'image/svg+xml' });
}

import sampleBoxUrl from '../assets/sample_box.jpg';
export { sampleBoxUrl };

export async function loadSampleBoxFile(): Promise<{ file: File; url: string; width: number; height: number }> {
  const res = await fetch(sampleBoxUrl);
  const blob = await res.blob();
  const file = new File([blob], 'sample_box_1.jpg', { type: 'image/jpeg' });
  return {
    file,
    url: sampleBoxUrl,
    width: 1024,
    height: 1024,
  };
}

