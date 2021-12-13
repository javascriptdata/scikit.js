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

import { describe, it } from 'mocha'
import { dfd } from '../../globals'
import { meanSquaredError } from '../metrics/metrics'
import { Tensor1D, Tensor2D } from '@tensorflow/tfjs'
import { KNeighborsRegressor } from './kNeighborsRegressor'
import { assert } from 'chai'
import { loadDiabetes } from '../datasets/datasets'

type int = number

// TODO: replace this with KFold as soon as its implemented
function* kFoldIndices(
  k: int,
  len: int
): IterableIterator<[Int32Array, Int32Array]> {
  if (!(0 < k && k % 1 === 0))
    throw new Error('kFoldIndices(k,len): k must be a positive integer.')
  k |= 0

  if (!(k <= len && len % 1 === 0))
    throw new Error('kFoldIndices(k,len): len must be an int greater than k.')
  len |= 0

  const n = (len / k) | 0
  let rem = len % k

  const range = new Int32Array(len)
  for (let i = len; i-- > 0; ) range[i] = i

  let off = 0
  for (let chunk: number; off < len; off += chunk) {
    chunk = n + +(rem-- > 0)
    const test = new Int32Array(len - chunk)
    test.set(range.subarray(0, off), 0)
    test.set(range.subarray(off + chunk), off)
    yield [test, range.slice(off, off + chunk)]
  }

  assert.equal(off, len)
}

function* kFold(
  k: int,
  X: Tensor2D,
  y: Tensor1D
): IterableIterator<[Tensor2D, Tensor1D, Tensor2D, Tensor1D]> {
  const len = X.shape[0]
  if (len !== y.shape[0])
    throw new Error('kFold(k,X,y): X.shape[0] must equal y.shape[0].')

  for (const [train, test] of kFoldIndices(k, len))
    yield [X.gather(train), y.gather(train), X.gather(test), y.gather(test)]
}

describe('KNeighborsRegressor', () => {
  const reference: [
    () => Promise<dfd.DataFrame>,
    { nNeighbors: int; weights: 'distance' | 'uniform' },
    int
  ][] = [
    [loadDiabetes, { nNeighbors: 5, weights: 'distance' }, 3570],
    [loadDiabetes, { nNeighbors: 3, weights: 'uniform' }, 3833]
  ]

  for (const [loadData, params, referenceError] of reference)
    it(`KNeighborsRegressor(${JSON.stringify(params)}) fits ${
      loadData.name
    } as well as sklearn`, async () => {
      const df = await loadData()

      const Xy = df.tensor as unknown as Tensor2D
      let [nSamples, nFeatures] = Xy.shape
      --nFeatures

      const X = Xy.slice([0, 0], [nSamples, nFeatures]),
        y = Xy.slice([0, nFeatures]).reshape<Tensor1D>([nSamples])

      const k = 3

      const model = new KNeighborsRegressor(params as any)

      let mse = 0

      for (const data of kFold(k, X, y))
        try {
          const [train_X, train_y, test_X, test_y] = data
          await model.fit(train_X, train_y)
          const predict_y = model.predict(test_X)
          mse += meanSquaredError(test_y, predict_y)
        } finally {
          for (const tensor of data) tensor.dispose()
        }

      mse /= k

      assert.isAtMost(mse, referenceError * 1.1)
    })
})
