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
import { Metric, minkowskiDistance } from './metrics'
import { Scikit1D, Scikit2D } from '../types'
import { Tensor1D } from '@tensorflow/tfjs'
import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'

/**
 * Common super-interface for {@ling KNeighborsRegressorParams}
 * and {@link KNeighborsClassifierParams}.
 */
export interface KNeighborsBaseParams {
  /**
   * The algorithms used to compute nearest neighbors.
   */
  algorithm?: 'auto' | 'brute'
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
  metric?: 'manhattan' | 'euclidean' | 'chebyshev' | 'minkowski' | Metric
  /**
   * The number of neighbors used during prediction.
   */
  nNeighbors?: number
}

/**
 * Common superclass for {@link KNeighborsRegressor} and {@link KNeighborsClassifier}.
 * Handles common constructor parameters and fitting.
 */
export class KNeighborsBase {
  private _metricFn: Metric
  private _createNeighborhood: (
    params: NeighborhoodParams
  ) => Promise<Neighborhood>

  protected _neighborhood: Neighborhood | undefined
  protected _y: Tensor1D | undefined

  private _algorithm: KNeighborsBaseParams['algorithm']
  private _p: KNeighborsBaseParams['p']
  private _metric: KNeighborsBaseParams['metric']

  nNeighbors: number

  get algorithm() {
    return this._algorithm
  }
  get p() {
    return this._p
  }
  get metric() {
    return this._metric
  }

  constructor({
    algorithm = 'auto',
    p = 2,
    metric = 'minkowski',
    nNeighbors = 5
  }: KNeighborsBaseParams) {
    this._algorithm = algorithm
    this._p = p
    this._metric = metric

    this.nNeighbors = Math.floor(nNeighbors)
    if (this.nNeighbors <= 0 || this.nNeighbors != nNeighbors) {
      throw new Error(
        `new KNeighborsRegressor({nNeighbors}): nNeighbors must be a positive integer.`
      )
    }

    switch (algorithm) {
      case 'auto':
        algorithm = 'brute'
        break
      case 'brute':
        break
      default:
        throw new Error('new KNeighborsBase({algorithm}): invalid algorithm.')
    }
    this._createNeighborhood = BruteNeighborhood.create

    switch (metric) {
      case 'minkowski':
        this._metricFn = minkowskiDistance(p)
        break
      case 'manhattan':
        this._metricFn = minkowskiDistance(1)
        this._p = 1
        break
      case 'euclidean':
        this._metricFn = minkowskiDistance(2)
        this._p = 2
        break
      case 'chebyshev':
        this._metricFn = minkowskiDistance(Infinity)
        this._p = Infinity
        break
      default:
        if ('function' !== typeof metric)
          throw new Error('new KNeighborsBase({metric}): invalid metric.')
        this._metricFn = metric
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
    const { _createNeighborhood, _metricFn: metric } = this
    const entries = convertToNumericTensor2D(X)
    this._neighborhood = await _createNeighborhood({ metric, entries })
    this._y = convertToNumericTensor1D(y)
  }
}
