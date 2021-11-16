import { Layer } from './base';
import nj, { NdArray } from '@d4c/numjs';
import { col2im, im2col } from '../utils/cnn';

export class Pooling implements Layer {
  poolH: number;
  poolW: number;
  stride: number;
  pad: number;
  argMax: NdArray = nj.zeros(0);
  xBatch: NdArray = nj.zeros(0);

  constructor(poolH: number, poolW: number, stride = 1, pad = 0) {
    this.poolH = poolH;
    this.poolW = poolW;
    this.stride = stride;
    this.pad = pad;
  }

  forward(): void {
    return;
  }
  forwardBatch(xBatch: NdArray): NdArray {
    const [N, C, H, W] = xBatch.shape;
    const outH = Math.floor(1 + (H - this.poolH) / this.stride);
    const outW = Math.floor(1 + (W - this.poolW) / this.stride);
    const col = im2col(
      xBatch,
      this.poolH,
      this.poolW,
      this.stride,
      this.pad
    ).reshape(outH * outW * N * C, this.poolH * this.poolW) as NdArray;
    const out = (
      nj
        .array(
          col.tolist().map((convoluteOutput) => {
            return Math.max(...convoluteOutput);
          })
        )
        .reshape(N, outH, outW, C) as NdArray
    ).transpose(0, 3, 1, 2);
    this.argMax = nj.array(
      col.tolist().map((convoluteOutput) => {
        return convoluteOutput.findIndex(
          (item) => item === Math.max(...convoluteOutput)
        );
      })
    );
    this.xBatch = xBatch;
    return out;
  }
  backward(): void {
    return;
  }
  backwardBatch(dout: NdArray): NdArray {
    const poolSize = this.poolH * this.poolW;
    dout = dout.transpose(0, 2, 3, 1);
    let dmax = nj
      .zeros(dout.size * poolSize)
      .reshape(dout.size, poolSize) as NdArray;
    const doutFlattenLs = dout.flatten().tolist() as number[];
    for (let i = 0; i < dout.size; i++) {
      dmax.set(i, this.argMax.get(i), doutFlattenLs[i]);
    }
    dmax = dmax.reshape(...dout.shape, poolSize);
    const dcol = dmax.reshape(
      dmax.shape[0] * dmax.shape[1] * dmax.shape[2],
      -1
    ) as NdArray;
    const [n, d, h, w] = this.xBatch.shape;
    const dx = col2im(
      dcol,
      { n, d, h, w },
      this.poolH,
      this.poolW,
      this.stride,
      this.pad
    );
    return dx;
  }
}
