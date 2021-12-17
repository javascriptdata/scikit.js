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

import { Neighborhood, NeighborhoodParams } from './neighborhood'
import { BruteNeighborhood } from './bruteNeighborhood'
import { minkowskiDistance } from './metrics'
import { Scikit1D, Scikit2D } from '../types'
import { Tensor1D, Tensor2D } from '@tensorflow/tfjs'
import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
import { assert } from '../typesUtils'
import { tf } from '../../globals'

const WEIGHTS_FUNCTIONS = {
  uniform(distances: Tensor2D) {
    const { shape } = distances
    return tf.fill(shape, 1 / shape[1]) as Tensor2D
  },
  distance(distances: Tensor2D) {
    return tf.tidy(() => {
      // safeMin is the smallest float32 value whose reciprocal is finite, i.e.:
      // safeMin = min{ x: float32 | x > 0 && 1 / x < Infinity }
      const safeMin = 2.938737e-39

      // if there are distances of zero, we have to avoid division by zero
      let if0 = tf.less(distances, safeMin)
      const num0 = if0.sum(1, /*keepDims=*/ true)
      if0 = if0.toFloat().div(num0)

      // otherwise we can use the inverse distance base weighting
      let non0 = tf.div(1, distances)
      non0 = non0.div(non0.sum(1, /*keepDims=*/ true))

      return tf.where(num0.greater(0), if0, non0) as Tensor2D
    })
  }
}

const METRIC_FUNCTIONS = {
  minkowski: (p: number) => minkowskiDistance(p),
  manhattan: () => minkowskiDistance(1),
  euclidean: () => minkowskiDistance(2),
  chebyshev: () => minkowskiDistance(Infinity)
}

const ALGORITHMS = {
  auto: async (params: NeighborhoodParams) => new BruteNeighborhood(params),
  brute: async (params: NeighborhoodParams) => new BruteNeighborhood(params)
}

/**
 * Common super-interface for {@link KNeighborsRegressorParams}
 * and {@link KNeighborsClassifierParams}.
 */
export interface KNeighborsParams {
  /**
   * Weighting strategy for the neighboring target values during
   * prediction. `'uniform'` gives all targets the same weight
   * independent of distance. `'distance'` uses inverse distance
   * based weighting.
   */
  weights?: keyof typeof WEIGHTS_FUNCTIONS
  /**
   * The algorithms used to compute nearest neighbors.
   */
  algorithm?: keyof typeof ALGORITHMS
  /**
   * Power parameter for the Minkowski metric.
   * `p=1` corresponds to the `manhattan` distance.
   * `p=2` corresponds to the `euclidean` distance.
   * `p=Infinity` corresponds to the `chebyshev` distance.
   * </ul>
   */
  p?: number
  /**
   * The metric to be used to compute nearest neighbor distances.
   */
  metric?: keyof typeof METRIC_FUNCTIONS
  /**
   * The number of neighbors used during prediction.
   */
  nNeighbors?: number
}

/**
 * Common superclass for {@link KNeighborsRegressor} and {@link KNeighborsClassifier}.
 * Handles common constructor parameters and fitting.
 */
export class KNeighborsBase implements KNeighborsParams {
  private _neighborhood: Neighborhood | undefined
  private _y: Tensor1D | undefined

  weights: KNeighborsParams['weights']
  algorithm: KNeighborsParams['algorithm']
  p: KNeighborsParams['p']
  metric: KNeighborsParams['metric']
  nNeighbors: KNeighborsParams['nNeighbors']

  constructor(params: KNeighborsParams) {
    Object.assign(this, params)
  }

  protected _getFitParams() {
    const { _neighborhood, _y, nNeighbors = 5, weights = 'uniform' } = this
    assert(
      0 <= nNeighbors && nNeighbors % 1 === 0,
      'KNeighbors({nNeighbors})::predict(X): nNeighbors must be a positive int.'
    )
    assert(
      Object.prototype.hasOwnProperty.call(WEIGHTS_FUNCTIONS, weights),
      'KNeighbors({weights})::predict(X): invalid weights.'
    )
    assert(
      undefined != _neighborhood && undefined != _y,
      'KNeighbors::predict(X): model not trained yet.'
    )

    const weightsFn = WEIGHTS_FUNCTIONS[weights]

    // make sure TypeScript knows that neighborhood and y are not undefined
    return {
      nNeighbors,
      weightsFn,
      neighborhood: _neighborhood as Neighborhood,
      y: _y as Tensor1D
    }
  }

  /**
   * Async function. Trains this model using the given features and targets.
   *
   * @param X The features of each training sample, where `X[i,j]` is the
   *          (j+1)-th feature of (i+1)-th sample.
   * @param y The target of each training sample, where `y[i]` the the
   *          target of the (i+1)-th sample.
   */
  async fit(X: Scikit2D, y: Scikit1D) {
    const { algorithm = 'auto', metric = 'minkowski', p = 2 } = this
    assert(
      Object.prototype.hasOwnProperty.call(METRIC_FUNCTIONS, metric),
      'KNeighbors({metric}).fit(X,y): invalid metric.'
    )
    assert(
      Object.prototype.hasOwnProperty.call(ALGORITHMS, algorithm),
      'KNeighbors({algorithm}).fit(X,y): invalid algorithm.'
    )

    const metricFn = METRIC_FUNCTIONS[metric](p)

    const entries = convertToNumericTensor2D(X)
    this._neighborhood = await ALGORITHMS[algorithm]({
      entries,
      metric: metricFn
    })
    this._y = convertToNumericTensor1D(y)
  }
}
