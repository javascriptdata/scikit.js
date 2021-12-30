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

import { Scikit2D } from '../types'
import { KNeighborsBase } from '../neighbors/kNeighborsBase'
import { convertToNumericTensor2D } from '../utils'
import { tf } from '../shared/globals'

/**
 * K-Nearest neighbor regressor.
 *
 * @example
 * ```js
 * import {KNeighborsRegressor} from 'scikitjs'
 *
 * let X = [[0], [1], [2], [3]]
 * let y = [0, 0, 1, 1]
 *
 * let knn = new KNeighborsRegressor(nNeighbor)
 *
 * await knn.fit(X, y)
 *
 * knn.predict([[1.5]]).print()
 * ```
 */
export class KNeighborsRegressor extends KNeighborsBase {
  /**
   * Applies this mdodel to predicts the target of each given sample.
   *
   * @param X The samples for which the targets are to be predicted,
   *          where `X[i,j]` is the (j+1)-th feature of the (i+1)-th
   *          sample.
   * @param y The predicted targets `y` where `y[i]` is the prediction
   *          for sample `X[i,:]`
   */
  public predict(X: Scikit2D) {
    const { neighborhood, y, nNeighbors, weightsFn } = this._getFitParams()

    return tf.tidy(() => {
      const _X = convertToNumericTensor2D(X)
      const { distances, indices } = neighborhood.kNearest(nNeighbors, _X)

      const targets = y.gather(indices)
      const weights = weightsFn(distances)

      // return tf.einsum('ij,ij->i', targets, weights) as Tensor1D
      return tf
        .matMul(
          targets.reshape([-1, 1, nNeighbors]),
          weights.reshape([-1, nNeighbors, 1])
        )
        .as1D()
    })
  }
}
