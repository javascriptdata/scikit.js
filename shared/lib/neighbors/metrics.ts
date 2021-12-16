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

import { Tensor1D, Tensor2D } from '@tensorflow/tfjs'
import { tf } from '../../globals'

/**
 * Abstract type of distance metrics, which compute the
 * distances between a stack of points `X` and a single point `y`.
 *
 * @param X A 2d tensor where each row represents a point.
 * @param y A 1d tensor representing a single point.
 *
 * @returns A 1d tensor `dist` where `dist[i]` is the
 *          metric distance between `X[i,:]` and `y[:]`.
 */
export type Metric = (X: Tensor2D, y: Tensor1D) => Tensor1D

/**
 * Returns the Minkowski distance metric with the given power `p`.
 * It is equivalent to the p-norm of the absolute difference
 * between two vectors.
 *
 * @param p The power/exponent of the Minkowski distance.
 * @returns `(X,y) => sum[i]( |X[:,i]-y[i]|**p ) ** (1/p)`
 */
export const minkowskiDistance = (p: number) => (X: Tensor2D, y: Tensor1D) => {
  // FIXME: tf.norm still underflows and overflows,
  // see: https://github.com/tensorflow/tfjs/issues/895
  return tf.tidy(() => tf.norm(tf.sub(X, y), p, -1) as Tensor1D)
}
