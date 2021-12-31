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

import { Tensor2D } from '@tensorflow/tfjs-core'
import { tf } from '../shared/globals'

/**
 * Abstract type of distance metrics, which compute the
 * distances between a stack of points `X` and a single point `y`.
 *
 * @param X A 2d tensor where each row represents a point.
 * @param y A 2d tensor where each row represents a point.
 *
 * @returns A 2d tensor `dist` where `dist[i,j]` is the
 *          metric distance between `u[i,:]` and `v[j,:]`.
 */
export type Metric = (u: Tensor2D, v: Tensor2D) => Tensor2D

/**
 * Returns the Minkowski distance metric with the given power `p`.
 * It is equivalent to the p-norm of the absolute difference
 * between two vectors.
 *
 * @param p The power/exponent of the Minkowski distance.
 * @returns `(X,y) => sum[i]( |X[:,i]-y[i]|**p ) ** (1/p)`
 */
export const minkowskiDistance = (p: number) => (u: Tensor2D, v: Tensor2D) => {
  // FIXME: tf.norm still underflows and overflows,
  // see: https://github.com/tensorflow/tfjs/issues/895
  const [m, s] = u.shape
  const [n, t] = v.shape

  return tf.tidy(() => {
    const x = u.reshape([m, 1, s])
    const y = v.reshape([1, n, t])
    return tf.norm(tf.sub(x, y), p, -1)
  }) as Tensor2D
}
