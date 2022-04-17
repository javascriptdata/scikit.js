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
import { tf } from '../shared/globals'
import { assert } from '../typesUtils'

/**
 * A {@link Neighborhood} implementation that uses a brute force approach
 * to nearest neighbor search. During a {@link BruteNeighborhood#kNearest}
 * query, the distance between every entry and the query point is computed.
 */
export class BruteNeighborhood implements Neighborhood {
  private _metric: Metric
  private _entries: tf.Tensor2D

  constructor({ metric, entries }: NeighborhoodParams) {
    this._metric = metric
    this._entries = entries
  }

  kNearest(k: number, queryPoints: tf.Tensor2D) {
    const { _metric, _entries } = this

    assert(
      _entries.shape[1] == queryPoints.shape[1],
      'X_train.shape[1] must equal X_predict.shape[1]'
    )

    // // batched version
    // const [m, n] = queryPoints.shape
    // return tf.tidy(() => {
    //   const negDist = _metric.tensorDistance(
    //     queryPoints.reshape([m, 1, n]),
    //     _entries
    //   ).neg() as Tensor2D
    //   const { values, indices } = tf.topk(negDist, k)
    //   return { distances: values.neg(), indices }
    // })

    // unbatched version
    return tf.tidy(() => {
      const result = tf.unstack(queryPoints).map((queryPoint) => {
        return tf.tidy(() => {
          const dist = _metric.tensorDistance(queryPoint, _entries).neg()
          const { values, indices } = tf.topk(dist, k)
          return [values, indices]
        })
      })
      return {
        distances: tf.stack(result.map((x) => x[0])).neg() as tf.Tensor2D,
        indices: tf.stack(result.map((x) => x[1])) as tf.Tensor2D
      }
    })
  }
}
