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
import { Metric } from './metrics'
import { Tensor1D, Tensor2D } from '@tensorflow/tfjs'
import { tf } from '../../globals'

/**
 * A {@link Neighborhood} implementation that uses a brute force approach
 * to nearest neighbor search. During a {@link BruteNeighborhood#kNearest}
 * query, the distance between every entry and the query point is computed.
 */
export class BruteNeighborhood implements Neighborhood {
  private _metric: Metric
  private _entries: Tensor2D

  static async create({ metric, entries }: NeighborhoodParams) {
    const result = {
      _metric: metric,
      _entries: entries
    }
    Object.setPrototypeOf(result, BruteNeighborhood.prototype)
    return result as unknown as BruteNeighborhood
  }

  private constructor() {
    throw new Error('Cannot construct directly, use create instead.')
  }

  kNearest(k: number, address: Tensor1D): [Tensor1D, Tensor1D] {
    const { _metric, _entries } = this
    const distances = _metric(_entries, address).neg()
    const { values, indices } = tf.topk(distances, k)
    return [values.neg(), indices]
  }
}
