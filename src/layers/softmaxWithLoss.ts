import { Layer } from './base';
import nj, { NdArray } from '@d4c/numjs';
import { softmax, softmaxBatch } from '../utils/activation';
import { crossEntroyError, crossEntroyErrorBatch } from '../utils/loss';

export class SoftmaxWithLoss implements Layer {
  y: NdArray;
  yBatch: NdArray;
  t: NdArray;
  tBatch: NdArray;
  loss: number;
  constructor(
    y: NdArray = nj.zeros(0),
    yBatch: NdArray = nj.zeros(0),
    t: NdArray = nj.zeros(0),
    tBatch: NdArray = nj.zeros(0)
  ) {
    this.y = y;
    this.yBatch = yBatch;
    this.t = t;
    this.tBatch = tBatch;
    this.loss = -1;
  }
  forward(x: NdArray, t: NdArray): number {
    this.y = softmax(x);
    this.t = t;
    this.loss = crossEntroyError(this.y, this.t);
    return this.loss;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  backward(dout = 1): NdArray {
    return this.y.subtract(this.t);
  }
  forwardBatch(
    xBatch: NdArray,
    tBatch: NdArray
  ): number {
    this.yBatch = softmaxBatch(xBatch);
    this.tBatch = tBatch;
    this.loss = crossEntroyErrorBatch(this.yBatch, this.tBatch);
    return this.loss;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  backwardBatch(dout = 1): NdArray {
    const batchSize = this.tBatch.shape[0];
    return this.yBatch.subtract(this.tBatch).divide(batchSize);
  }
}
