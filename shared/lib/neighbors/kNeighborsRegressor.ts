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
import {
  KNeighborsBase,
  KNeighborsBaseParams
} from '../neighbors/kNeighborsBase'
import { convertToNumericTensor2D } from '../utils'
import { Tensor1D } from '@tensorflow/tfjs'
import { tf } from '../../globals'

type WeightsFn = (distances: Tensor1D) => Tensor1D

const WEIGHTS_FUNCTIONS = {
  uniform(distances: Tensor1D) {
    const { shape } = distances
    return tf.fill(shape, 1 / shape[0]) as Tensor1D
  },
  distance(distances: Tensor1D) {
    return tf.tidy(() => {
      // safeMin is the smallest float32 value whose reciprocal is finite, i.e.:
      // safeMin = min{ x: float32 | x > 0 && 1 / x < Infinity }
      const safeMin = 2.938737e-39

      // if there are distances of zero, we have to avoid division by zero
      let if0 = tf.less(distances, safeMin)
      const num0 = if0.sum()
      if0 = if0.asType('float32').div(num0)

      // otherwise we can use the inverse distance base weighting
      let non0 = tf.div(1, distances)
      non0 = non0.div(non0.sum())

      return tf.where(num0.greater(0), if0, non0) as Tensor1D
    })
  }
}

export interface KNeighborsRegressorParams extends KNeighborsBaseParams {
  /**
   * Weighting strategy for the neighboring target values during
   * prediction. `'uniform'` gives all targets the same weight
   * independent of distance. `'distance'` uses inverse distance
   * based weighting.
   */
  weights?: WeightsFn | keyof typeof WEIGHTS_FUNCTIONS
}

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
 * knn.predict([[1.5]]).print()
 * ```
 */
export class KNeighborsRegressor extends KNeighborsBase {
  private _weightsFn: WeightsFn

  constructor(params: KNeighborsRegressorParams = {}) {
    super(params)
    const { weights = 'uniform' } = params

    if ('function' === typeof weights) {
      this._weightsFn = weights
    } else if (
      Object.prototype.hasOwnProperty.call(WEIGHTS_FUNCTIONS, weights)
    ) {
      this._weightsFn =
        WEIGHTS_FUNCTIONS[weights as keyof typeof WEIGHTS_FUNCTIONS]
    } else {
      throw new Error('new KNeighborsRegressor({weights}): unknown weights.')
    }
  }

  /**
   * Applies this mdodel to predicts the target of each given sample.
   *
   * @param X The samples for which the targets are to be predicted,
   *          where `X[i,j]` is the (j+1)-th feature of the (i+1)-th
   *          sample.
   * @param y The predicted targets `y` where `y[i]` is the prediction
   *          for sample `X[i,:]`
   */
  predict(X: Scikit2D) {
    const { nNeighbors, _weightsFn, _neighborhood, _y } = this
    if (null == _neighborhood || null == _y) {
      throw new Error(
        'KNeighborsRegressor::predict(X): model not trained yet.'
      )
    }

    return tf.tidy(() => {
      const _X = convertToNumericTensor2D(X)

      return tf.stack(
        tf.unstack(_X).map((x) => {
          const [dist, indices] = _neighborhood.kNearest(
            nNeighbors,
            x as Tensor1D
          )
          const weights = _weightsFn(dist)
          return _y.gather(indices).dot(weights)
        })
      ) as Tensor1D
    })
  }
}
