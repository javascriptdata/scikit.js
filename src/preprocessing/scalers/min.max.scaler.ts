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

import { Tensor1D, tensor1d } from '@tensorflow/tfjs-node'
import {
  convertToNumericTensor1D_2D,
  convertTensorToInputType
} from '../../utils'
import { ScikitVecOrMatrix } from '../../types'
import { isScikitVecOrMatrix, assert } from '../../types.utils'
import { tensorMin, tensorMax, turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'
/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 */

export default class MinMaxScaler extends TransformerMixin {
  $scale: Tensor1D
  $min: Tensor1D

  constructor() {
    super()
    this.$scale = tensor1d([])
    this.$min = tensor1d([])
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
  fit(data: ScikitVecOrMatrix) {
    assert(
      isScikitVecOrMatrix(data),
      'Data can not be converted to a 1D or 2D matrix.'
    )

    const tensorArray = convertToNumericTensor1D_2D(data)
    const max = tensorMax(tensorArray, 0, true) as Tensor1D
    this.$min = tensorMin(tensorArray, 0, true) as Tensor1D
    let scale = max.sub(this.$min)

    // But what happens if max = min, ie.. we are dealing with a constant vector?
    // In the case above, scale = max - min = 0 and we'll divide by 0 which is no bueno.
    // The common practice in cases where the vector is constant is to change the 0 elements
    // in scale to 1, so that the division doesn't fail. We do that below
    this.$scale = turnZerosToOnes(scale) as Tensor1D

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
  transform(data: ScikitVecOrMatrix) {
    assert(
      isScikitVecOrMatrix(data),
      'Data can not be converted to a 1D or 2D matrix.'
    )
    const tensorArray = convertToNumericTensor1D_2D(data)
    const outputData = tensorArray.sub(this.$min).div(this.$scale)
    return convertTensorToInputType(outputData, data)
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
  inverseTransform(data: ScikitVecOrMatrix) {
    assert(
      isScikitVecOrMatrix(data),
      'Data can not be converted to a 1D or 2D matrix.'
    )
    const tensorArray = convertToNumericTensor1D_2D(data)
    const outputData = tensorArray.mul(this.$scale).add(this.$min)
    return convertTensorToInputType(outputData, data)
  }
}
