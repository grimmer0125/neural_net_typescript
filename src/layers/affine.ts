import { Layer } from './base';
import nj, { NdArray } from '@d4c/numjs';

export class Affine implements Layer {
  W: NdArray;
  b: NdArray;
  dW: NdArray;
  db: NdArray;
  x: NdArray;
  xBatch: NdArray;
  xShape: number[];
  constructor(
    W: NdArray,
    b: NdArray,
    x: NdArray = nj.zeros(0),
    xBatch: NdArray = nj.zeros(0)
  ) {
    this.W = W;
    this.b = b;
    this.dW = nj.zeros(0);
    this.db = nj.zeros(0);
    this.x = x;
    this.xBatch = xBatch;
    this.xShape = xBatch.shape;
  }

  /*
    forwardは X・W + b を返す。
  */
  forward(x: NdArray): NdArray {
    this.x = x;
    const xMat = x.reshape(1, x.size) as NdArray;
    const bMat = this.b.reshape(1, this.b.size) as NdArray;
    return nj.add(nj.dot(xMat, this.W), bMat).flatten();
  }

  forwardBatch(
    xBatch: NdArray
  ): NdArray {
    this.xShape = xBatch.shape;
    this.xBatch = xBatch.reshape(this.xShape[0], -1) as NdArray;
    const batchSize = xBatch.shape[0];
    const bMat = this.b.reshape(1, this.b.size) as NdArray;
    const ones = nj.ones(batchSize).reshape(batchSize, 1) as NdArray;
    const bMatAdd = nj.dot(ones, bMat);
    return nj.dot(this.xBatch, this.W).add(bMatAdd);
  }

  backward(dout: NdArray): NdArray {
    this.db = dout;
    this.dW = nj.dot(
      this.x.reshape(this.x.size, 1) as NdArray,
      dout.reshape(1, dout.size) as NdArray
    );
    return nj
      .dot(dout.reshape(1, dout.size) as NdArray, this.W.T)
      .flatten();
  }

  backwardBatch(
    dout: NdArray
  ): NdArray {
    const batchSize = this.xBatch.shape[0];
    this.db = nj
      .dot(
        dout.T,
        nj.ones(batchSize).reshape(batchSize, 1) as NdArray
      )
      .flatten();
    this.dW = nj.dot(this.xBatch.T, dout);
    const dx = nj.dot(dout, this.W.T);
    return dx.reshape(...this.xShape);
  }
}
