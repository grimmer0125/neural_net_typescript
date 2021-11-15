import { Layer } from './base';
import nj, { NdArray } from '@d4c/numjs';

export class Relu implements Layer {
  mask: NdArray;
  maskBatch: NdArray;

  constructor(
    mask: NdArray = nj.zeros(0),
    maskBatch: NdArray = nj.zeros(0)
  ) {
    this.mask = mask;
    this.maskBatch = maskBatch;
  }

  forward = (x: NdArray): NdArray => {
    const xArray = x.tolist();
    this.mask = nj.array(xArray.map((xItem) => Number(xItem > 0)));
    return x.multiply(this.mask);
  };

  forwardBatch = (xBatch: NdArray): NdArray => {
    const xArrayBatch = xBatch.tolist();
    this.maskBatch = nj.array(
      xArrayBatch.map((xArray) => xArray.map((x) => Number(x > 0)))
    );
    return xBatch.multiply(this.maskBatch);
  };

  backward = (dout: NdArray): NdArray => {
    return dout.multiply(this.mask);
  };

  backwardBatch = (dout: NdArray): NdArray => {
    return dout.multiply(this.maskBatch);
  };
}
