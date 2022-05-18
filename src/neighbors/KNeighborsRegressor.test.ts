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

import {
  KNeighborsRegressor,
  setBackend,
  KFold,
  crossValScore
} from '../index'
import { KNeighborsParams } from './KNeighborsBase'
import { dataUrls } from '../datasets/datasets'
import { arrayEqual } from '../utils'
import { negMeanSquaredError } from '../model_selection/scorers'
import '../jestTensorMatchers'
import * as dfd from 'danfojs-node'
import { Tensor1D, Tensor2D } from '../types'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

function testWithDataset(
  loadDataUrl: string,
  loadDataName: string,
  params: KNeighborsParams,
  referenceError: number
) {
  it(
    `matches sklearn fitting ${loadDataName}`.padEnd(48) +
      JSON.stringify(params),
    async () => {
      const df = await dfd.readCSV(loadDataUrl)

      const Xy = df.tensor as unknown as Tensor2D
      let [nSamples, nFeatures] = Xy.shape
      --nFeatures

      const X = Xy.slice([0, 0], [nSamples, nFeatures])
      const y = Xy.slice([0, nFeatures]).reshape([nSamples]) as Tensor1D

      const scores = await (crossValScore as any)(
        new KNeighborsRegressor(params),
        X,
        y,
        {
          cv: new KFold({ nSplits: 3 }),
          scoring: negMeanSquaredError
        }
      )

      expect(scores.mean()).toBeAllCloseTo(-referenceError, {
        atol: 0,
        rtol: 0.005
      })
    },
    90_000
  )
}

for (const algorithm of [
  ...KNeighborsRegressor.SUPPORTED_ALGORITHMS,
  undefined
]) {
  describe(`KNeighborsRegressor({ algorithm: ${algorithm} })`, function () {
    testWithDataset(
      dataUrls.loadDiabetes,
      'loadDiabetes',
      { nNeighbors: 5, weights: 'distance', algorithm },
      3570
    )
    testWithDataset(
      dataUrls.loadDiabetes,
      'loadDiabetes',
      { nNeighbors: 3, weights: 'uniform', algorithm },
      3833
    )
    if ('kdTree' === algorithm) {
      testWithDataset(
        dataUrls.fetchCaliforniaHousing,
        'fetchCaliforniaHousing',
        { nNeighbors: 3, weights: 'distance', algorithm },
        1.31
      )
    }
    if ('auto' === algorithm) {
      testWithDataset(
        dataUrls.fetchCaliforniaHousing,
        'fetchCaliforniaHousing',
        { nNeighbors: 4, weights: 'uniform', algorithm, p: 1 },
        1.19
      )
    }
    if (undefined === algorithm) {
      testWithDataset(
        dataUrls.fetchCaliforniaHousing,
        'fetchCaliforniaHousing',
        { nNeighbors: 4, weights: 'uniform', algorithm, p: Infinity },
        1.32
      )
    }

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
      expect(
        arrayEqual(knn.predict(predictX).arraySync(), [20, 20, 20], 0.01)
      ).toBe(true)
    }, 60_000)
  })
}
