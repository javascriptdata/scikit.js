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

import { Tensor2D } from '@tensorflow/tfjs'
import { Metric } from './metrics'

/**
 * Default constructor parameters for {@link Neighborhood} instances.
 */
export interface NeighborhoodParams {
  /**
   * Distance metric used for neighborhood queries.
   */
  metric: Metric
  /**
   * A 2d tensor containing the entries of the neighborhood.
   * The row `entries[i,:]` represents the (i+1)-th point.
   * The nearest neighbors are searched for in these points.
   */
  entries: Tensor2D
}

/**
 * A collections of float vectors that allows (reasonably) fast
 * nearest neighbor according to some distance metric.
 */
export interface Neighborhood {
  /**
   * Returns the k nearest neighbors from this {@link Neighborhood}.
   *
   * @param k The number of nearest neighbors to be returned.
   * @param queryPoints Query point to which the nearest neighbors are
   *                    to be searched.
   *
   * @returns `[dist, indices]` where `dist` are the distances from `address`
   *          to its `k` nearest neighbors. `indices` are the indices
   *          of the `k` nearest neighbors in the {@link Neighborhood}'s
   *          {@link NeighborhoodParams#entries | entries}.
   */
  kNearest(
    k: number,
    queryPoints: Tensor2D
  ): { distances: Tensor2D; indices: Tensor2D }
}
