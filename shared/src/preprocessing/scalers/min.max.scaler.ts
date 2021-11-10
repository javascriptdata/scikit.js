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

import { convertToNumericTensor2D } from '../../utils'
import { Scikit2D, Transformer } from '../../types'
import { isScikit2D, assert } from '../../types.utils'
import { tensorMin, tensorMax, turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'

import { tf } from '../../globals'
/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 */

export default class MinMaxScaler
  extends TransformerMixin
  implements Transformer
{
  $scale: tf.Tensor1D
  $min: tf.Tensor1D

  constructor() {
    super()
    this.$scale = tf.tensor1d([])
    this.$min = tf.tensor1d([])
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
  fit(X: Scikit2D): MinMaxScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')

    const tensorArray = convertToNumericTensor2D(X)
    const max = tensorMax(tensorArray, 0, true) as tf.Tensor1D
    this.$min = tensorMin(tensorArray, 0, true) as tf.Tensor1D
    let scale = max.sub(this.$min)

    // But what happens if max = min, ie.. we are dealing with a constant vector?
    // In the case above, scale = max - min = 0 and we'll divide by 0 which is no bueno.
    // The common practice in cases where the vector is constant is to change the 0 elements
    // in scale to 1, so that the division doesn't fail. We do that below
    this.$scale = turnZerosToOnes(scale) as tf.Tensor1D

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
  transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.sub(this.$min).div<tf.Tensor2D>(this.$scale)
    return outputData
  }

  /**
   * Inverse transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.inverseTransform([0, 0.25, 0.5, 0.75, 1])
   * // [1, 2, 3, 4, 5]
   * */
  inverseTransform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.mul(this.$scale).add<tf.Tensor2D>(this.$min)
    return outputData
  }
}
