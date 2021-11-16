import { Layer } from './base';
import nj, { NdArray } from '@d4c/numjs';

/*
 画像などの3次元データのためのRelu関数を表すクラス。ConvolutionとPoolingの間に入る層として用いる。
*/
export class ImageRelu implements Layer {
  maskBatch: NdArray = nj.zeros(0);

  forward(): void {
    return;
  }
  forwardBatch(xBatch: NdArray): NdArray {
    const xImageArray = xBatch.tolist();
    this.maskBatch = nj.array(
      xImageArray.map((xImage) =>
        xImage.map((xChannel) =>
          xChannel.map((xArray) => xArray.map((x) => Number(x > 0)))
        )
      )
    );
    return xBatch.multiply(this.maskBatch);
  }

  backward(): void {
    return;
  }

  backwardBatch(dout: NdArray): NdArray {
    return dout.multiply(this.maskBatch);
  }
}
