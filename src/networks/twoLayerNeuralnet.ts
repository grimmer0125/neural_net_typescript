import nj, { NdArray } from '@d4c/numjs';
import { Layer } from '../layers/base';
import { Affine } from '../layers/affine';
import { Relu } from '../layers/relu';
import { SoftmaxWithLoss } from '../layers/softmaxWithLoss';
import { softmax, softmaxBatch } from '../utils/activation';

export class TwoLayerNet {
  W1: NdArray;
  b1: NdArray;
  W2: NdArray;
  b2: NdArray;
  layers: Layer[];
  lossLayer: Layer;
  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    this.W1 = nj
      .random([inputSize * hiddenSize])
      .multiply(0.01)
      .reshape(inputSize, hiddenSize) as NdArray;
    this.b1 = nj.zeros([hiddenSize]);
    this.W2 = nj
      .random([hiddenSize * outputSize])
      .multiply(0.01)
      .reshape(hiddenSize, outputSize) as NdArray;
    this.b2 = nj.zeros([outputSize]);
    this.layers = [
      new Affine(this.W1, this.b1),
      new Relu(),
      new Affine(this.W2, this.b2),
    ];
    this.lossLayer = new SoftmaxWithLoss();
  }

  predict(x: NdArray): NdArray {
    let output = x;
    for (const layer of this.layers) {
      output = layer.forward(output);
    }
    return softmax(output);
  }

  predictBatch(xBatch: NdArray): NdArray {
    let output = xBatch;
    for (const layer of this.layers) {
      output = layer.forwardBatch(output);
    }
    return softmaxBatch(output);
  }

  /* 基本的にはバッチ学習であることを前提とする。 */
  forward(xBatch: NdArray, tBatch: NdArray): number {
    let scoreBatch: NdArray = xBatch;
    for (const layer of this.layers) {
      scoreBatch = layer.forwardBatch(scoreBatch);
    }
    const loss = this.lossLayer.forwardBatch(scoreBatch, tBatch);
    return loss;
  }
  backward(): void {
    let dout: NdArray = this.lossLayer.backwardBatch();
    const reversedLayers = this.layers.slice().reverse();
    for (const layer of reversedLayers) {
      dout = layer.backwardBatch(dout);
    }
  }
  update(learningRate = 0.1): void {
    const { dW1, db1, dW2, db2 } = this.gradient();
    this.W1 = this.W1.subtract(dW1.multiply(learningRate));
    this.b1 = this.b1.subtract(db1.multiply(learningRate));
    this.W2 = this.W2.subtract(dW2.multiply(learningRate));
    this.b2 = this.b2.subtract(db2.multiply(learningRate));
    (this.layers[0] as Affine).W = this.W1;
    (this.layers[0] as Affine).b = this.b1;
    (this.layers[2] as Affine).W = this.W2;
    (this.layers[2] as Affine).b = this.b2;
  }

  gradient(): {
    dW1: NdArray;
    db1: NdArray;
    dW2: NdArray;
    db2: NdArray;
  } {
    const affine1 = this.layers[0] as Affine;
    const affine2 = this.layers[2] as Affine;
    return {
      dW1: affine1.dW,
      db1: affine1.db,
      dW2: affine2.dW,
      db2: affine2.db,
    };
  }
}
