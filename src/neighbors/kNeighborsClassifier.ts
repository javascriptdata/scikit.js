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

import { Scikit1D, Scikit2D } from '../types'
import { KNeighborsBase } from './kNeighborsBase'
import { convertToNumericTensor2D, convertToTensor1D } from '../utils'
import { tf } from '../shared/globals'
import { Tensor1D, Tensor2D } from '@tensorflow/tfjs'
import { polyfillUnique } from '../tfUtils'

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
export class KNeighborsClassifier extends KNeighborsBase {
  classes_?: Tensor1D

  /**
   * Applies this mdodel to predict the class probabilities of each given sample.
   *
   * @param X The samples for which the targets are to be predicted,
   *          where `X[i,j]` is the (j+1)-th feature of the (i+1)-th
   *          sample.
   * @param Y The predicted class probabilities `Y` where `Y[i,j]` is the
   *          predicted probability of sample `X[i,:]` having the to belong
   *          to class with index `j`.
   */
  public predictProba(X: Scikit2D): Tensor2D {
    const { neighborhood, y, nNeighbors, weightsFn } = this._getFitParams()
    const [nClasses] = this.classes_?.shape as [number]

    return tf.tidy(() => {
      const _X = convertToNumericTensor2D(X)
      const nSamples = _X.shape[0]
      const { distances, indices } = neighborhood.kNearest(nNeighbors, _X)

      const labels = y.gather(indices)
      const weight = weightsFn(distances)
      const oneHot = tf.oneHot(labels, nClasses)

      return tf
        .mul(
          oneHot.reshape([nSamples, nNeighbors, nClasses]),
          weight.reshape([nSamples, nNeighbors, 1])
        )
        .sum(1) as Tensor2D
    })
  }

  /**
   * Applies this mdodel to predict the class of each given sample.
   *
   * @param X The samples for which the targets are to be predicted,
   *          where `X[i,j]` is the (j+1)-th feature of the (i+1)-th
   *          sample.
   * @param y The predicted targets `y` where `y[i]` is the prediction
   *          for sample `X[i,:]`
   */
  public predict(X: Scikit2D): Tensor1D {
    const classes = this.classes_ as Tensor1D

    return tf.tidy(() => {
      const probs = this.predictProba(X)
      const labels = probs.argMax(1)
      return classes.gather(labels)
    })
  }

  public async fit(X: Scikit2D, labels: Scikit1D): Promise<this> {
    const { values, indices } = tf.tidy(() => {
      const _labels = convertToTensor1D(labels)
      polyfillUnique(tf)
      return tf.unique(_labels)
    })
    super.fit(X, indices)
    this.classes_ = values
    return this
  }
}
