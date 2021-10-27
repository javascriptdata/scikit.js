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

import { abs, Tensor1D, tensor1d, Tensor2D } from '@tensorflow/tfjs-node'
import { convertToNumericTensor2D } from '../../utils'
import { Scikit2D, Transformer } from '../../types'
import { isScikit2D, assert } from '../../types.utils'
import { tensorMin, tensorMax, turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'
/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 */

export default class Normalizer extends TransformerMixin {
  norm: string
  constructor(norm = 'l2') {
    super()
    this.norm = norm
  }

  /**
   * Fits a MinMaxScaler to the data
   * @param data Array, Tensor, DataFrame or Series object
   * @returns MinMaxScaler
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * // MinMaxScaler {
   * //   $max: [5],
   * //   $min: [1]
   * // }
   *
   */
  fit(X: Scikit2D): Normalizer {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    return this
  }

  /**
   * Transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.transform([1, 2, 3, 4, 5])
   * // [0, 0.25, 0.5, 0.75, 1]
   * */
  transform(X: Scikit2D): Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    if (this.norm === 'l1') {
      const means = abs(tensorArray).sum(1).reshape([-1, 1])
      return tensorArray.div(means)
    }
    if (this.norm === 'l2') {
      const means = tensorArray.square().sum(1).sqrt().reshape([-1, 1])
      return tensorArray.div(means)
    }
    // max case
    const means = abs(tensorArray).max(1).reshape([-1, 1])
    return tensorArray.div(means)
  }
}
