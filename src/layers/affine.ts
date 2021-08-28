import { Layer } from './base';
import nj from 'numjs';

export class Affine implements Layer {
  W: nj.NdArray<number[]>;
  b: nj.NdArray<number>;
  dW: nj.NdArray<number[]>;
  db: nj.NdArray<number>;
  x: nj.NdArray<number>;
  xBatch: nj.NdArray<number[]>;
  constructor(W: nj.NdArray<number[]>, b: nj.NdArray<number>) {
    this.W = W;
    this.b = b;
    this.dW = nj.zeros(0);
    this.db = nj.zeros(0);
    this.x = nj.zeros(0);
    this.xBatch = nj.zeros(0);
  }
  forward(x: nj.NdArray<number>): nj.NdArray<number> {
    const xMat = x.reshape(1, x.size) as nj.NdArray<number[]>;
    const bMat = this.b.reshape(1, this.b.size) as nj.NdArray<number[]>;
    return nj.add(nj.dot(xMat, this.W), bMat).flatten();
  }
  forwardBatch(xBatch: nj.NdArray<number[]>): nj.NdArray<number[]> {
    const batchSize = xBatch.shape[0];
    const bMat = this.b.reshape(1, this.b.size) as nj.NdArray<number[]>;
    const ones = nj.ones(batchSize).reshape(batchSize, 1) as nj.NdArray<
      number[]
    >;
    const bMatAdd = nj.dot(ones, bMat);
    return nj.dot(xBatch, this.W).add(bMatAdd);
  }

  backward(): void {
    1 + 1;
  }
  backwardBatch(): void {
    1 + 1;
  }
}
