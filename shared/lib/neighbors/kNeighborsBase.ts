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
  Neighborhood,
  NeighborhoodEntry,
  NeighborhoodParams
} from './neighborhood'
import { BruteNeighborhood } from './bruteNeighborhood'
import { Metric, minkowskiDistance } from './metrics'
import { Scikit1D, Scikit2D } from '../types'
import { Tensor1D } from '@tensorflow/tfjs'
import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
import { tf } from '../../globals'

type int = number

export interface KNeighborsBaseParams {
  algorithm?: 'auto' | 'brute'
  p?: number
  metric?:
    | 'manhattan'
    | 'euclidean'
    | 'chebyshev'
    | 'minkowski'
    | Metric<ArrayLike<number>>
}

export class KNeighborsBase {
  private _metric: Metric<ArrayLike<number>>
  private _NeighborhoodConstructor: new (
    params: NeighborhoodParams<ArrayLike<number>, int>
  ) => Neighborhood<ArrayLike<number>, int>

  protected _neighborhood: Neighborhood<ArrayLike<number>, int> | undefined
  protected _y: Tensor1D | undefined

  constructor({
    algorithm = 'auto',
    p = 2,
    metric = 'minkowski'
  }: KNeighborsBaseParams) {
    switch (algorithm) {
      case 'auto':
        algorithm = 'brute'
        break
      case 'brute':
        break
      default:
        throw new Error('new KNeighborsBase({algorithm}): invalid algorithm.')
    }
    this._NeighborhoodConstructor = BruteNeighborhood

    switch (metric) {
      case 'minkowski':
        this._metric = minkowskiDistance(p)
        break
      case 'manhattan':
        this._metric = minkowskiDistance(1)
        break
      case 'euclidean':
        this._metric = minkowskiDistance(2)
        break
      case 'chebyshev':
        this._metric = minkowskiDistance(Infinity)
        break
      default:
        if ('function' !== typeof metric)
          throw new Error('new KNeighborsBase({algorithm}): invalid metric.')
        this._metric = metric
    }
  }

  protected async _fit(_X: Scikit2D, _y: Scikit1D) {
    const { _NeighborhoodConstructor, _metric: metric } = this

    tf.engine().startScope()
    try {
      const X = convertToNumericTensor2D(_X),
        y = convertToNumericTensor1D(_y)

      const l = y.shape[0],
        [m, n] = X.shape

      if (l !== m)
        throw new Error(
          'KNeighborsBase::_fit(X,y): X.shape[0] must equal y.shape[0]'
        )

      const addr = await X.data()

      const entries: NeighborhoodEntry<ArrayLike<number>, int>[] = []

      for (let i = 0; i < m; )
        entries.push({
          value: i,
          address: addr.slice(i * n, ++i * n)
        })

      this._neighborhood = new _NeighborhoodConstructor({ metric, entries })
      this._y = y
    } finally {
      tf.engine().endScope(this._y)
    }
  }
}
