/**
*  @license
* Copyright 2021, JsData. All rights reserved.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==========================================================================
*/


import { dfd, tf } from '../shared/globals'
import { meanSquaredError } from '../metrics/metrics'
import { Tensor1D, Tensor2D } from '@tensorflow/tfjs-core'
import { KNeighborsRegressor } from './kNeighborsRegressor'
import { KNeighborsParams } from './kNeighborsBase'
import { fetchCaliforniaHousing, loadDiabetes } from '../datasets/datasets'
import { arrayEqual } from '../utils'

// TODO: replace this with KFold as soon as its implemented
function* kFoldIndices(
  k: number,
  len: number
): IterableIterator<[Int32Array, Int32Array]> {
  if (!(0 < k && k % 1 === 0)) {
    throw new Error('kFoldIndices(k,len): k must be a positive integer.')
  }
  k = Math.floor(k)

  if (!(k <= len && len % 1 === 0)) {
    throw new Error('kFoldIndices(k,len): len must be an int greater than k.')
  }
  len = Math.floor(len)

  const n = Math.floor(len / k)
  let rem = len % k

  const range = new Int32Array(len)
  for (let i = len; i-- > 0; ) {
    range[i] = i
  }

  let off = 0
  for (let chunk: number; off < len; off += chunk) {
    chunk = n + Number(rem-- > 0)

    const test = new Int32Array(len - chunk)
    test.set(range.subarray(0, off), 0)
    test.set(range.subarray(off + chunk), off)

    yield [test, range.slice(off, off + chunk)]
  }

  expect(off).toEqual(len)
}

function* kFold(
  k: number,
  X: Tensor2D,
  y: Tensor1D
): IterableIterator<[Tensor2D, Tensor1D, Tensor2D, Tensor1D]> {
  const len = X.shape[0]
  if (len !== y.shape[0]) {
    throw new Error('kFold(k,X,y): X.shape[0] must equal y.shape[0].')
  }

  for (const [train, test] of kFoldIndices(k, len)) {
    yield [X.gather(train), y.gather(train), X.gather(test), y.gather(test)]
  }
}

function testWithDataset(
  loadData: () => Promise<dfd.DataFrame>,
  params: KNeighborsParams,
  referenceError: number
) {
  it(`KNeighborsRegressor(${JSON.stringify(params)}) fits ${
    loadData.name
  } as well as sklearn`, async () => {
    const df = await loadData()

    const Xy = df.tensor as unknown as Tensor2D
    let [nSamples, nFeatures] = Xy.shape
    --nFeatures

    const X = Xy.slice([0, 0], [nSamples, nFeatures])
    const y = Xy.slice([0, nFeatures]).reshape([nSamples]) as Tensor1D

    const k = 3

    const model = new KNeighborsRegressor(params)

    let mse = 0

    for (const data of kFold(k, X, y)) {
      try {
        tf.engine().startScope()
        const [train_X, train_y, test_X, test_y] = data
        await model.fit(train_X, train_y)
        const predict_y = model.predict(test_X)
        mse += meanSquaredError(test_y, predict_y)
      } finally {
        tf.engine().endScope()
        for (const tensor of data) tensor.dispose()
      }
    }

    mse /= k

    expect(Math.abs(mse - referenceError)).toBeLessThanOrEqual(Math.abs(referenceError) * 0.02)
  })
}

for (const algorithm of [
  'kdTree',
  'brute',
  undefined,
  'auto'
] as KNeighborsParams['algorithm'][]) {
  describe(`KNeighborsRegressor({ algorithm: ${algorithm} })`, function () {
    // testWithDataset(
    //   loadDiabetes,
    //   { nNeighbors: 5, weights: 'distance', algorithm },
    //   3570
    // )
    // testWithDataset(
    //   loadDiabetes,
    //   { nNeighbors: 3, weights: 'uniform', algorithm },
    //   3833
    // )
    // testWithDataset(
    //   fetchCaliforniaHousing,
    //   { nNeighbors: 3, weights: 'distance', algorithm },
    //   1.31
    // )
    // testWithDataset(
    //   fetchCaliforniaHousing,
    //   { nNeighbors: 4, weights: 'uniform', algorithm },
    //   1.28
    // )
    // testWithDataset(
    //   fetchCaliforniaHousing,
    //   { nNeighbors: 4, weights: 'uniform', algorithm, p: 1 },
    //   1.19
    // )
    // testWithDataset(
    //   fetchCaliforniaHousing,
    //   { nNeighbors: 4, weights: 'uniform', algorithm, p: Infinity },
    //   1.32
    // )

    it('correctly predicts sklearn example', async () => {
      const X = [[0], [1], [2], [3]]
      const y = [0, 0, 1, 1]

      const model = new KNeighborsRegressor({ algorithm, nNeighbors: 2 })

      await model.fit(X, y)

      expect(model.predict([[1.5]]).arraySync()).toEqual([0.5])
    }, 60_000)
    // test cases as suggested by @dcrescim
    it('Use KNeighborsRegressor on simple example (n=1)', async function () {
      const knn = new KNeighborsRegressor({ algorithm, nNeighbors: 1 })

      const X = [
        [-1, 0],
        [0, 0],
        [5, 0]
      ]
      const y = [10, 20, 30]
      const predictX = [
        [1, 0],
        [4, 0],
        [-5, 0]
      ]

      await knn.fit(X, y)
      expect(knn.predict(predictX).arraySync()).toEqual([20, 30, 10])
    }, 60_000)
    it('Use KNeighborsRegressor on simple example (n=2)', async function () {
      const knn = new KNeighborsRegressor({ algorithm, nNeighbors: 2 })

      const X = [
        [-1, 0],
        [0, 0],
        [5, 0]
      ]
      const y = [10, 20, 30]
      const predictX = [
        [1, 0],
        [4, 0],
        [-5, 0]
      ]

      await knn.fit(X, y)
      expect(knn.predict(predictX).arraySync()).toEqual([15, 25, 15])
    }, 60_000)
    it('Use KNeighborsRegressor on simple example (n=3)', async function () {
      const knn = new KNeighborsRegressor({ algorithm, nNeighbors: 3 })

      const X = [
        [-1, 0],
        [0, 0],
        [5, 0]
      ]
      const y = [10, 20, 30]
      const predictX = [
        [1, 0],
        [4, 0],
        [-5, 0]
      ]

      await knn.fit(X, y)
      expect(arrayEqual(knn.predict(predictX).arraySync(), [20, 20, 20], 0.01)).toBe(true)
    }, 60_000)
  })
}
