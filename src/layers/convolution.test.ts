import { Convolution } from './convolution';
import nj, { NdArray } from '@d4c/numjs';

describe('Convolution Layer Test', () => {
  describe('Convolution,forward', () => {
    test('forward', () => {
      const W = nj.ones([1, 1, 3, 3]).reshape(1, 1, 3, 3) as NdArray;
      const xBatch = nj.ones([1, 1, 5, 5]).reshape(1, 1, 5, 5) as NdArray; // 10枚 の 3*5*5次元の画像
      const b = nj.zeros(1);
      const conv = new Convolution(W, b);
      expect(conv.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [9, 9, 9],
            [9, 9, 9],
            [9, 9, 9],
          ],
        ],
      ]);
    });
    test('forward', () => {
      const W = nj.ones([5, 3, 3, 3]).reshape(5, 3, 3, 3) as NdArray;
      const xBatch = nj.ones([10, 3, 5, 5]).reshape(10, 3, 5, 5) as NdArray; // 10枚 の 3*5*5次元の画像
      const b = nj.arange(5);
      const conv = new Convolution(W, b);
      expect(conv.forwardBatch(xBatch).tolist()).toEqual([
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
        [
          [
            [27, 27, 27],
            [27, 27, 27],
            [27, 27, 27],
          ],
          [
            [28, 28, 28],
            [28, 28, 28],
            [28, 28, 28],
          ],
          [
            [29, 29, 29],
            [29, 29, 29],
            [29, 29, 29],
          ],
          [
            [30, 30, 30],
            [30, 30, 30],
            [30, 30, 30],
          ],
          [
            [31, 31, 31],
            [31, 31, 31],
            [31, 31, 31],
          ],
        ],
      ]);
    });
  });
  describe('Convolution,backward', () => {
    test('backward', () => {
      const W = nj.ones([1, 1, 3, 3]).reshape(1, 1, 3, 3) as NdArray;
      const xBatch = nj.ones([1, 1, 5, 5]).reshape(1, 1, 5, 5) as NdArray; // 10枚 の 3*5*5次元の画像
      const b = nj.zeros(1);
      const conv = new Convolution(W, b);
      conv.forwardBatch(xBatch);
      const dout = nj.ones([1, 3, 3, 1]).reshape(1, 3, 3, 1) as NdArray;
      expect(conv.backwardBatch(dout).tolist()).toEqual([
        [
          [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 9, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1],
          ],
        ],
      ]);
    });
  });
});
