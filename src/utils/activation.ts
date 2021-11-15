import nj, { NdArray } from '@d4c/numjs';

export const softmax = (x: NdArray): NdArray => {
  x = x.add(-x.max());
  return nj.divide(nj.exp(x), x.exp().sum());
};

export const softmaxBatch = (
  xBatch: NdArray
): NdArray => {
  return nj.array(xBatch.tolist().map((x) => softmax(nj.array(x)).tolist()));
};
