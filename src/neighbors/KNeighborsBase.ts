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

import { Neighborhood, NeighborhoodParams } from './Neighborhood'
import { BruteNeighborhood } from './BruteNeighborhood'
import { minkowskiMetric } from './Metric'
import { Scikit1D, Scikit2D, Tensor2D, Tensor1D } from '../types'
import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
import { assert } from '../typesUtils'
import { KdTree } from './KdTree'
import Serialize from '../serialize'
import { getBackend } from '../tf-singleton'

const WEIGHTS_FUNCTIONS = {
  uniform(distances: Tensor2D) {
    let tf = getBackend()
    const { shape } = distances
    return tf.fill(shape, 1 / shape[1]) as Tensor2D
  },
  distance(distances: Tensor2D) {
    let tf = getBackend()
    return tf.tidy(() => {
      // scale inverse distances by min. to avoid `1/tinyVal == Infinity`
      const min = distances.min(1, /*keepDims=*/ true)
      const invDist = tf.divNoNan(min.toFloat(), distances)

      const is0 = distances.lessEqual(0).toFloat()

      // avoid div by 0 by using `1/0 == 1` and `1/(x!=0) == 0` instead
      const weights = tf.where(min.lessEqual(0), is0, invDist)
      const wsum = weights.sum(1, /*keepDims=*/ true)

      return weights.div(wsum)
    })
  }
}

const METRICS = {
  minkowski: (p: number) => minkowskiMetric(p),
  manhattan: () => minkowskiMetric(1),
  euclidean: () => minkowskiMetric(2),
  chebyshev: () => minkowskiMetric(Infinity)
}

const ALGORITHMS = {
  kdTree: KdTree.build,
  brute: async (params: NeighborhoodParams) => new BruteNeighborhood(params),
  auto: (params: NeighborhoodParams) => {
    return 'function' === typeof params.metric.minDistToBBox
      ? ALGORITHMS.kdTree(params)
      : ALGORITHMS.brute(params)
  }
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
   * For tree-based algorithms, this is a hint as to how many
   * points are to be stored in a single leaf. The optimal
   * depends on the nature of the problem and the metric used.
   */
  leafSize?: number
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
  metric?: keyof typeof METRICS
  /**
   * The number of neighbors used during prediction.
   */
  nNeighbors?: number
}

/**
 * Common superclass for {@link KNeighborsRegressor} and {@link KNeighborsClassifier}.
 * Handles common constructor parameters and fitting.
 */
export class KNeighborsBase extends Serialize implements KNeighborsParams {
  static readonly SUPPORTED_ALGORITHMS = Object.freeze(
    Object.keys(ALGORITHMS)
  ) as (keyof typeof ALGORITHMS)[]

  _neighborhood: Neighborhood | undefined
  _y: Tensor1D | undefined

  weights: KNeighborsParams['weights']
  algorithm: KNeighborsParams['algorithm']
  leafSize: KNeighborsParams['leafSize']
  p: KNeighborsParams['p']
  metric: KNeighborsParams['metric']
  nNeighbors: KNeighborsParams['nNeighbors']

  constructor(params: KNeighborsParams = {}) {
    super()
    Object.assign(this, params)
  }

  protected _getFitParams() {
    const { _neighborhood, _y, nNeighbors = 5, weights = 'uniform' } = this
    assert(
      0 <= nNeighbors && nNeighbors % 1 === 0,
      'KNeighbors({nNeighbors})::predict(X): nNeighbors must be a positive int.'
    )
    assert(
      Object.keys(WEIGHTS_FUNCTIONS).includes(weights),
      'KNeighbors({weights})::predict(X): invalid weights.'
    )
    assert(
      undefined != _neighborhood && undefined != _y,
      'KNeighbors::predict(X): model not trained yet. Call `await fit(x, y)` first.'
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
  public async fit(X: Scikit2D, y: Scikit1D): Promise<this> {
    const { algorithm = 'auto', metric = 'minkowski', p = 2, leafSize } = this
    assert(
      Object.keys(METRICS).includes(metric),
      'KNeighbors({metric}).fit(X,y): invalid metric.'
    )
    assert(
      Object.keys(ALGORITHMS).includes(algorithm),
      'KNeighbors({algorithm}).fit(X,y): invalid algorithm.'
    )

    const metricFn = METRICS[metric](p)

    const entries = convertToNumericTensor2D(X)
    this._neighborhood = await ALGORITHMS[algorithm]({
      entries,
      metric: metricFn,
      leafSize
    })
    this._y = convertToNumericTensor1D(y)
    return this
  }
}
