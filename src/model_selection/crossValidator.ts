/**
*  @license
* Copyright 2022, JsData. All rights reserved.
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
import { tf } from '../shared/globals'
type Tensor1D = tf.Tensor1D

/**
 * Interface for cross validation splitting strategies.
 */
export interface CrossValidator {
  /**
   * Returns the number of splits this {@link CrossValidator}
   * would produce for the given dataset.
   *
   * @param X Feature tensor.
   * @param y Target tensor (optional).
   * @param groups A tensor containing grouping information.
   *               Used by some CrossValidators to make sure
   *               that data of the same group does not appear
   *               both in the test and training dataset, e.g.
   *               multiple blood samples from a single individual.
   */
  getNumSplits(X: Scikit2D, y?: Scikit1D, groups?: Scikit1D): number
  /**
   * Yields the different splits into training and test data.
   *
   * @param X Feature tensor.
   * @param y Target tensor (optional).
   * @param groups A tensor containing grouping information.
   *               Used by some CrossValidators to make sure
   *               that data of the same group does not appear
   *               both in the test and training dataset, e.g.
   *               multiple blood samples from a single individual.
   * @yields Trainings splits, where `trainIndex` represents the
   *         indices belonging to training data and `testIndex`
   *         the ones belonging to the test data.
   */
  split(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  ): IterableIterator<{ trainIndex: Tensor1D; testIndex: Tensor1D }>
}
