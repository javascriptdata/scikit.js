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
import { Metric } from './Metric'
import { assert } from '../typesUtils'
import { Tensor2D } from '../types'
import { getBackend } from '../tf-singleton'
/**
 * A {@link Neighborhood} implementation that uses a brute force approach
 * to nearest neighbor search. During a {@link BruteNeighborhood#kNearest}
 * query, the distance between every entry and the query point is computed.
 */
export class BruteNeighborhood implements Neighborhood {
  _metric: Metric
  _entries: Tensor2D
  tf: any

  constructor({ metric, entries }: NeighborhoodParams) {
    this._metric = metric
    this._entries = entries
    this.tf = getBackend()
  }

  kNearest(k: number, queryPoints: Tensor2D) {
    const { _metric, _entries } = this

    assert(
      _entries.shape[1] == queryPoints.shape[1],
      'X_train.shape[1] must equal X_predict.shape[1]'
    )

    // // batched version
    // const [m, n] = queryPoints.shape
    // return this.tf.tidy(() => {
    //   const negDist = _metric.tensorDistance(
    //     queryPoints.reshape([m, 1, n]),
    //     _entries
    //   ).neg() as Tensor2D
    //   const { values, indices } = this.tf.topk(negDist, k)
    //   return { distances: values.neg(), indices }
    // })

    // unbatched version
    return this.tf.tidy(() => {
      const result = this.tf.unstack(queryPoints).map((queryPoint: any) => {
        return this.tf.tidy(() => {
          const dist = _metric.tensorDistance(queryPoint, _entries).neg()
          const { values, indices } = this.tf.topk(dist, k)
          return [values, indices]
        })
      })
      return {
        distances: this.tf
          .stack(result.map((x: any) => x[0]))
          .neg() as Tensor2D,
        indices: this.tf.stack(result.map((x: any) => x[1])) as Tensor2D
      }
    })
  }
}
