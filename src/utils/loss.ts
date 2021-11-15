import nj, { NdArray } from '@d4c/numjs';

export const crossEntroyError = (
  y: NdArray,
  t: NdArray
): number => {
  const logY = nj.array(y.tolist().map((yItem) => Math.log(yItem + 1e-7)));
  return -nj.sum(nj.multiply(t, logY));
};

export const crossEntroyErrorBatch = (
  yBatch: NdArray,
  tBatch: NdArray
): number => {
  const batchSize = tBatch.shape[0];
  const logYBatch = nj.array(
    yBatch.tolist().map((y) => y.map((yItem) => Math.log(yItem + 1e-7)))
  );
  return -nj.sum(nj.multiply(tBatch, logYBatch)) / batchSize;
};
