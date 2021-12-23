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
import { KNeighborsClassifier } from './kNeighborsClassifier'
import { assert } from 'chai'
import { arrayEqual } from '../utils'

describe('KNeighborsClassifier', async () => {

  it('correctly predicts sklearn example', async () => {
    const X_train = [[0], [1], [2], [3]]
    const y_train = [0, 0, 1, 1]

    const model = new KNeighborsClassifier({ nNeighbors: 3 })
    await model.fit(X_train, y_train)

    const prob = await model.predictProba([[0.9]]).array()
    const pred = await model.predict([[1.1]]).array()

    assert.isTrue(arrayEqual(prob, [[2 / 3, 1 / 3]], 0.01))
    assert.deepEqual(pred, [0])
  })
  it('correctly predicts 1d example classes', async () => {
    const X_train = [[2], [0], [1], [7]]
    const y_train = [1.25, -1.75, 5.5, 1337]

    const model = new KNeighborsClassifier({
      nNeighbors: 3,
      weights: 'distance'
    })
    await model.fit(X_train, y_train)

    const X_test = [[0], [800], [1.1], [1.9]]
    const y_test = [-1.75, 1337, 5.5, 1.25]

    for (let i = 0; i < X_test.length; i++) {
      assert.deepEqual(await model.predict([X_test[i]]).array(), [y_test[i]])
    }
    assert.deepEqual(await model.predict(X_test).array(), y_test)
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
      assert.deepEqual(await model.predictProba([X_test[i]]).array(), [
        p_test[i]
      ])
    }
    assert.deepEqual(await model.predictProba(X_test).array(), p_test)
  })
})
