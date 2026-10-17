import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeAffineMatrix,
  mapViewportToImage,
  calculateCropCoordinates,
  computeMatrixCrop,
} from './matrix.ts';

test('computeAffineMatrix generates proper 3x3 forward and inverse matrices', () => {
  const params = {
    scale: 2,
    translateX: -100,
    translateY: -50,
    viewportW: 400,
    viewportH: 300,
  };

  const { matrix, inverseMatrix, determinant } = computeAffineMatrix(params);

  assert.equal(determinant, 4);

  assert.deepEqual(matrix, [
    [2, 0, -100],
    [0, 2, -50],
    [0, 0, 1],
  ]);

  assert.deepEqual(inverseMatrix, [
    [0.5, 0, 50],
    [0, 0.5, 25],
    [0, 0, 1],
  ]);

  const originImagePos = mapViewportToImage(0, 0, inverseMatrix);
  assert.equal(originImagePos.x, 50);
  assert.equal(originImagePos.y, 25);
});

test('calculateCropCoordinates calculates correct crop rectangle without clamp when inside image', () => {
  const params = {
    scale: 2,
    translateX: -100,
    translateY: -50,
    viewportW: 400,
    viewportH: 300,
  };

  const crop = calculateCropCoordinates(params, 1000, 800);

  assert.equal(crop.rawLeft, 50);
  assert.equal(crop.rawTop, 25);
  assert.equal(crop.rawWidth, 200);
  assert.equal(crop.rawHeight, 150);
  assert.equal(crop.left, 50);
  assert.equal(crop.top, 25);
  assert.equal(crop.width, 200);
  assert.equal(crop.height, 150);
  assert.equal(crop.isClamped, false);
});

test('calculateCropCoordinates clamps safely when user pans outside left/top boundary', () => {
  const params = {
    scale: 1,
    translateX: 50,
    translateY: 40,
    viewportW: 400,
    viewportH: 300,
  };

  const crop = calculateCropCoordinates(params, 800, 600);

  assert.equal(crop.left, 0);
  assert.equal(crop.top, 0);
  assert.equal(crop.isClamped, true);
  assert.ok(crop.width > 0 && crop.width <= 800);
  assert.ok(crop.height > 0 && crop.height <= 600);
});

test('calculateCropCoordinates clamps when viewport extends past bottom-right of image', () => {
  const params = {
    scale: 0.5,
    translateX: 0,
    translateY: 0,
    viewportW: 400,
    viewportH: 300,
  };

  const crop = calculateCropCoordinates(params, 500, 400);

  assert.equal(crop.rawWidth, 800);
  assert.equal(crop.rawHeight, 600);
  assert.equal(crop.left, 0);
  assert.equal(crop.top, 0);
  assert.equal(crop.width, 500);
  assert.equal(crop.height, 400);
  assert.equal(crop.isClamped, true);
});

test('computeMatrixCrop integrates both matrix and crop correctly', () => {
  const result = computeMatrixCrop(
    { scale: 1, translateX: 0, translateY: 0, viewportW: 400, viewportH: 300 },
    1200,
    900
  );

  assert.ok(result.affine.matrix);
  assert.ok(result.affine.inverseMatrix);
  assert.equal(result.crop.left, 0);
  assert.equal(result.crop.top, 0);
  assert.equal(result.crop.width, 400);
  assert.equal(result.crop.height, 300);
});
