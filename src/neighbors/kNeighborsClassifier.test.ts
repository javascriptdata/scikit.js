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

import { KNeighborsClassifier } from './kNeighborsClassifier'
import { KNeighborsParams } from './kNeighborsBase'
import {
  loadDigits,
  loadIris,
  loadWine,
  loadBreastCancer
} from '../datasets/datasets'
import { crossValScore } from '../model_selection/crossValScore'
import { KFold } from '../model_selection/kFold'
import { arrayEqual } from '../utils'
import '../jestTensorMatchers'
import * as dfd from 'danfojs'
import { tf } from '../shared/globals'
type Tensor1D = tf.Tensor1D
type Tensor2D = tf.Tensor2D

function testWithDataset(
  loadData: () => string,
  params: KNeighborsParams,
  referenceAccuracy: number
) {
  it(
    `matches sklearn fitting ${loadData.name}`.padEnd(48) +
      JSON.stringify(params),
    async () => {
      const df = await dfd.readCSV(loadData())

      const Xy = df.tensor as unknown as Tensor2D
      let [nSamples, nFeatures] = Xy.shape
      --nFeatures

      const X = Xy.slice([0, 0], [nSamples, nFeatures])
      const y = Xy.slice([0, nFeatures]).reshape([nSamples]) as Tensor1D

      const accuracies = await crossValScore(
        new KNeighborsClassifier(params),
        X,
        y,
        {
          cv: new KFold({ nSplits: 3 })
        }
      )

      expect(accuracies.mean()).toBeAllCloseTo(referenceAccuracy, {
        atol: 0,
        rtol: 0.005
      })
    },
    60_000
  )
}

for (const algorithm of [
  ...KNeighborsClassifier.SUPPORTED_ALGORITHMS,
  undefined
]) {
  describe(`KNeighborsClassifier({ algorithm: ${algorithm} })`, () => {
    testWithDataset(
      loadIris,
      { nNeighbors: 5, weights: 'distance', algorithm },
      0.0
    )
    testWithDataset(
      loadIris,
      { nNeighbors: 3, weights: 'uniform', algorithm },
      0.0
    )

    testWithDataset(
      loadWine,
      { nNeighbors: 5, weights: 'distance', algorithm },
      0.135
    )
    testWithDataset(
      loadWine,
      { nNeighbors: 3, weights: 'uniform', algorithm },
      0.158
    )

    testWithDataset(
      loadBreastCancer,
      { nNeighbors: 5, weights: 'distance', algorithm },
      0.92
    )
    testWithDataset(
      loadBreastCancer,
      { nNeighbors: 3, weights: 'uniform', algorithm },
      0.916
    )

    if ('brute' !== algorithm) {
      testWithDataset(
        loadDigits,
        { nNeighbors: 5, weights: 'distance', algorithm, leafSize: 256 },
        0.963
      )
      testWithDataset(
        loadDigits,
        { nNeighbors: 3, weights: 'uniform', algorithm, leafSize: 256 },
        0.967
      )
    }

    it('correctly predicts sklearn example', async () => {
      const X_train = [[0], [1], [2], [3]]
      const y_train = [0, 0, 1, 1]

      const model = new KNeighborsClassifier({ nNeighbors: 3, algorithm })
      await model.fit(X_train, y_train)

      const prob = await model.predictProba([[0.9]]).array()
      const pred = await model.predict([[1.1]]).array()

      expect(arrayEqual(prob, [[2 / 3, 1 / 3]], 0.01)).toBe(true)
      expect(pred).toEqual([0])
    })
    it('correctly predicts 1d example classes', async () => {
      const X_train = [[2], [0], [1], [7]]
      const y_train = [1.25, -1.75, 5.5, 1337]

      const model = new KNeighborsClassifier({
        algorithm,
        nNeighbors: 3,
        weights: 'distance'
      })
      await model.fit(X_train, y_train)

      const X_test = [[0], [800], [1.1], [1.9]]
      const y_test = [-1.75, 1337, 5.5, 1.25]

      for (let i = 0; i < X_test.length; i++) {
        expect(await model.predict([X_test[i]]).array()).toEqual([y_test[i]])
      }
      expect(await model.predict(X_test).array()).toEqual(y_test)
    })
    it('correctly predicts 2d example probabilities', async () => {
      const grid = [
        [1, 1, 2, 3, 3],
        [1, 1, 2, 3, 3],
        [2, 2, 2, 3, 3],
        [2, 2, 2, 3, 1]
      ]

      const X_train: number[][] = []
      const y_train: number[] = []

      for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[i].length; j++) {
          X_train.push([i, j])
          y_train.push(grid[i][j])
        }
      }

      const model = new KNeighborsClassifier({
        algorithm,
        nNeighbors: 4,
        weights: 'uniform'
      })
      await model.fit(X_train, y_train)

      const X_test = [
        [0.5, 0.5],
        [0.5, 1.5],
        [0.5, 2.5],
        [0.5, 3.5],
        [1.5, 0.5],
        [1.5, 1.5],
        [1.5, 2.5],
        [1.5, 3.5],
        [2.5, 0.5],
        [2.5, 1.5],
        [2.5, 2.5],
        [2.5, 3.5]
      ]
      const p_test = [
        [1, 0, 0],
        [0.5, 0.5, 0],
        [0, 0.5, 0.5],
        [0, 0, 1],
        [0.5, 0.5, 0],
        [0.25, 0.75, 0],
        [0, 0.5, 0.5],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0.25, 0, 0.75]
      ]

      for (let i = 0; i < X_test.length; i++) {
        expect(await model.predictProba([X_test[i]]).array()).toEqual([
          p_test[i]
        ])
      }
      expect(await model.predictProba(X_test).array()).toEqual(p_test)
    })
  })
}
