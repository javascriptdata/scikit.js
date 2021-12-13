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
import {
  KNeighborsBase,
  KNeighborsBaseParams
} from '../neighbors/kNeighborsBase'
import { convertToNumericTensor1D_2D } from '../utils'
import { Scalar, Tensor1D } from '@tensorflow/tfjs'
import { tf } from '../../globals'

type int = number
type WeightsFn = (distances: number[]) => ArrayLike<number>

const WEIGHTS_FUNCTIONS = {
  uniform(distances: number[]) {
    distances.fill(1 / distances.length)
    return distances
  },
  distance(distances: number[]) {
    const len = distances.length

    const nInf = distances.reduce((nInf, x) => nInf + +(x === Infinity), 0)

    if (nInf === 0) distances.fill(1 / len)
    else
      for (let i = len; i-- > 0; )
        distances[i] = +(distances[i] === Infinity) / nInf

    return distances
  }
  //  softmax( distances: number[] )
  //  {
  //    const len = distances.length
  //
  //    const min = distances.reduce( (x, y) => Math.min(x, y) )
  //
  //    let sum = 0
  //    for (let i = len; i-- > 0;) {
  //      const di = Math.exp(min - distances[i])
  //      sum += di
  //      distances[i] = di
  //    }
  //
  //    for (let i = len; i-- > 0;)
  //      distances[i] /= sum
  //
  //    return distances
  //  }
}

/**
 * Wraps a weightFn with input and ouput checks.
 *
 * @param weightsFn The wrapped weight function.
 * @returns `weightsFn` wrapped with checks/assertions.
 */
const checkedWeightsFn = (weightsFn: WeightsFn) => (distances: number[]) => {
  if (distances.some((d) => isNaN(d) || d < 0))
    throw new Error(
      'KNeighborsRegressor: metric function returned invalid distance.'
    )

  const weights = weightsFn(distances)

  if (weights.length !== distances.length)
    throw new Error(
      'KNeighborsRegressor: array returned by weights must have the same length than the input.'
    )

  let sum = 0
  for (let i = weights.length; i-- > 0; ) {
    const wi = weights[i]
    if (!(0 <= wi && wi <= 1))
      // <- handles NaN
      throw new Error(
        'KNeighborsRegressor: entries returned by weights must be in range [0,1].'
      )
    sum += wi
  }
  if (!(Math.abs(sum - 1) <= 1e-8))
    // <- handles NaN
    throw new Error(
      'KNeighborsRegressor: entries returned by weights must sum up to 1.'
    )

  return weights
}

export interface KNeighborsRegressorParams extends KNeighborsBaseParams {
  nNeighbors?: int
  weights?: WeightsFn | keyof typeof WEIGHTS_FUNCTIONS
}

export class KNeighborsRegressor extends KNeighborsBase {
  private _weightsFn: WeightsFn
  private _nNeighbors: int

  constructor(params: KNeighborsRegressorParams = {}) {
    super(params)
    const { nNeighbors = 5, weights = 'uniform' } = params

    this._nNeighbors = nNeighbors | 0
    if (this._nNeighbors <= 0 || this._nNeighbors != nNeighbors)
      throw new Error(
        `new KNeighborsRegressor({nNeighbors}): nNeighbors must be a positive integer.`
      )

    if ('function' === typeof weights)
      this._weightsFn = checkedWeightsFn(weights)
    // <- TODO: remove check after testing
    else if (Object.prototype.hasOwnProperty.call(WEIGHTS_FUNCTIONS, weights))
      this._weightsFn = checkedWeightsFn(
        WEIGHTS_FUNCTIONS[weights as keyof typeof WEIGHTS_FUNCTIONS]
      )
    else
      throw new Error(
        'new KNeighborsRegressor({weights}): weights must be of type ' +
          Object.keys(WEIGHTS_FUNCTIONS).join(' | ') +
          ' | (distance: number[n <= nNeighbors]) => number[n].'
      )
  }

  async fit(X: Scikit2D, y: Scikit1D) {
    await super._fit(X, y)
    return this
  }

  predict<T extends Scikit1D | Scikit2D>(
    _X: T
  ): T extends Scikit1D
    ? Scalar
    : T extends Scikit2D
    ? Tensor1D
    : Scalar | Tensor1D {
    const { _nNeighbors, _weightsFn, _neighborhood, _y } = this
    if (null == _neighborhood || null == _y)
      throw new Error(
        'KNeighborsRegressor::predict(X): model not trained yet.'
      )

    const X = convertToNumericTensor1D_2D(_X)
    if (X.rank === 2)
      return tf.stack(
        tf.unstack(X).map((x) => this.predict(x as Tensor1D))
      ) as any

    const dist: number[] = [],
      vals: int[] = []

    let n = 0
    for (const { distance, value } of _neighborhood.nearest(X.dataSync())) {
      dist.push(distance)
      vals.push(value)
      if (++n >= _nNeighbors) break
    }

    const weights = _weightsFn(dist)

    return _y.gather(vals).mul(weights).sum()
  }
}
