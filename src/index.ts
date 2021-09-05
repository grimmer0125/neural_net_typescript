import { TwoLayerNet } from './twoLayerNeuralnet';
import { loadMnist } from './data/load-mnist';
import { choice, getBatchData, maxIdx } from './utils';

const main = async () => {
  const { xTrain, yTrain, yTest, xTest } = await loadMnist();

  const network = new TwoLayerNet(784, 50, 10);
  const trainNum = xTrain.shape[0];
  const batchSize = 100;
  const iterNums = 1000;
  const learningRate = 0.1;
  const start = Date.now();
  for (let i = 0; i < iterNums; i++) {
    const batchIdxList = choice(trainNum, batchSize);
    const xBatch = getBatchData(batchIdxList, xTrain);
    const yBatch = getBatchData(batchIdxList, yTrain);
    const loss = network.forward(xBatch, yBatch);
    network.backward(); // 逆伝搬
    network.update(learningRate); // パラメータの更新
    if (i % 100 === 0) {
      const now = Date.now();
      console.log(now - start); // 実行にかかる時間を表示.
      console.log(loss); // 誤差関数の出力結果を表示.
    }
  }
  // テストデータを対象に精度を計算する。
  const testNum = yTest.shape[0];
  const testBatchIdxList = choice(testNum, 1000); // 1000個のデータで精度を検証する。
  const xTestBatch = getBatchData(testBatchIdxList, xTest);
  const yTestBatch = getBatchData(testBatchIdxList, yTest);
  const yPredict = network.predictBatch(xTestBatch);
  const yTestBatchList = yTestBatch.tolist();
  const yPredictList = yPredict.tolist();
  let accurateNum = 0;
  yTestBatchList.map((_, idx) => {
    accurateNum += Number(
      maxIdx(yTestBatchList[idx]) === maxIdx(yPredictList[idx])
    );
  });
  console.log(
    `Accuracy for 1000 of  test data : ${(accurateNum / 1000) * 100} % `
  );
};

main();
