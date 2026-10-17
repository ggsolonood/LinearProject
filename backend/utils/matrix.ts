/**
 * 2D Affine Transformation & Inversion Matrix Math
 * Backend Core Calculations
 */

export interface AffineMatrix3x3 {
  matrix: [[number, number, number], [number, number, number], [number, number, number]];
  inverseMatrix: [[number, number, number], [number, number, number], [number, number, number]];
  determinant: number;
}

export interface TransformParams {
  scale: number;
  translateX: number;
  translateY: number;
  viewportW: number;
  viewportH: number;
}

export interface CropCoordinates {
  rawLeft: number;
  rawTop: number;
  rawWidth: number;
  rawHeight: number;
  left: number;
  top: number;
  width: number;
  height: number;
  isClamped: boolean;
}

export interface MatrixCropResult {
  affine: AffineMatrix3x3;
  crop: CropCoordinates;
}

/**
 * Calculates the forward 3x3 Affine Matrix and its Inverse Matrix.
 */
export function computeAffineMatrix(params: TransformParams): AffineMatrix3x3 {
  const { scale, translateX, translateY } = params;
  const s = scale === 0 ? 0.0001 : scale;

  const matrix: [[number, number, number], [number, number, number], [number, number, number]] = [
    [s, 0, translateX],
    [0, s, translateY],
    [0, 0, 1],
  ];

  const determinant = s * s;

  const invS = 1 / s;
  const inverseMatrix: [[number, number, number], [number, number, number], [number, number, number]] = [
    [invS, 0, -translateX * invS],
    [0, invS, -translateY * invS],
    [0, 0, 1],
  ];

  return {
    matrix,
    inverseMatrix,
    determinant,
  };
}

/**
 * Map a point from viewport coordinate space (xv, yv) to original image space (ximg, yimg)
 */
export function mapViewportToImage(
  xv: number,
  yv: number,
  inverseMatrix: [[number, number, number], [number, number, number], [number, number, number]]
): { x: number; y: number } {
  const x = inverseMatrix[0][0] * xv + inverseMatrix[0][1] * yv + inverseMatrix[0][2];
  const y = inverseMatrix[1][0] * xv + inverseMatrix[1][1] * yv + inverseMatrix[1][2];
  return { x, y };
}

/**
 * Calculates the exact crop rectangle on the original image from the viewport,
 * clamping against image boundaries to prevent Sharp extraction errors.
 */
export function calculateCropCoordinates(
  params: TransformParams,
  imageWidth: number,
  imageHeight: number
): CropCoordinates {
  const { scale, translateX, translateY, viewportW, viewportH } = params;
  const s = Math.max(0.001, scale);

  const rawLeft = Math.round(-translateX / s);
  const rawTop = Math.round(-translateY / s);
  const rawWidth = Math.round(viewportW / s);
  const rawHeight = Math.round(viewportH / s);

  let left = Math.max(0, Math.min(rawLeft, imageWidth - 1));
  let top = Math.max(0, Math.min(rawTop, imageHeight - 1));

  let width = Math.min(rawWidth, imageWidth - left);
  let height = Math.min(rawHeight, imageHeight - top);

  if (rawLeft < 0) {
    left = 0;
    width = Math.min(imageWidth, Math.round((viewportW + translateX) / s));
  }
  if (rawTop < 0) {
    top = 0;
    height = Math.min(imageHeight, Math.round((viewportH + translateY) / s));
  }

  width = Math.max(1, Math.min(width, imageWidth - left));
  height = Math.max(1, Math.min(height, imageHeight - top));

  const isClamped =
    left !== rawLeft ||
    top !== rawTop ||
    width !== rawWidth ||
    height !== rawHeight;

  return {
    rawLeft,
    rawTop,
    rawWidth,
    rawHeight,
    left,
    top,
    width,
    height,
    isClamped,
  };
}

export function computeMatrixCrop(
  params: TransformParams,
  imageWidth: number,
  imageHeight: number
): MatrixCropResult {
  const affine = computeAffineMatrix(params);
  const crop = calculateCropCoordinates(params, imageWidth, imageHeight);
  return { affine, crop };
}
