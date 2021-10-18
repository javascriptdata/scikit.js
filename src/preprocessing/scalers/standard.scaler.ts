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

import { tensor1d, Tensor, Tensor1D } from '@tensorflow/tfjs-node'
import {
  convertTensorToInputType,
  convertToNumericTensor1D_2D,
  meanIgnoreNaN,
  turnZerosToOnes,
  stdIgnoreNaN,
} from '../../utils'
import { ScikitVecOrMatrix } from 'types'

/**
 * Standardize features by removing the mean and scaling to unit variance.
 * The standard score of a sample x is calculated as: `z = (x - u) / s`,
 * where `u` is the mean of the training samples, and `s` is the standard deviation of the training samples.
 */
export default class StandardScaler {
  $std: Tensor
  $mean: Tensor

  constructor() {
    this.$std = tensor1d([])
    this.$mean = tensor1d([])
  }

  /**
   * Fit a StandardScaler to the data.
   * @param data Array, Tensor, DataFrame or Series object
   * @returns StandardScaler
   * @example
   * const scaler = new StandardScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   */
  fit(data: ScikitVecOrMatrix) {
    const tensorArray = convertToNumericTensor1D_2D(data)
    const std = stdIgnoreNaN(tensorArray, 0)
    this.$mean = meanIgnoreNaN(tensorArray, 0)

    // Deal with zero variance issues
    this.$std = turnZerosToOnes(std) as Tensor1D
    return this
  }

  /**
   * Transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new StandardScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.transform([1, 2, 3, 4, 5])
   * // [0.0, 0.0, 0.0, 0.0, 0.0]
   * */
  transform(data: ScikitVecOrMatrix) {
    const tensorArray = convertToNumericTensor1D_2D(data)
    const outputData = tensorArray.sub(this.$mean).div(this.$std)
    return convertTensorToInputType(outputData, data)
  }

  /**
   * Fit and transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new StandardScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.fitTransform([1, 2, 3, 4, 5])
   * // [0.0, 0.0, 0.0, 0.0, 0.0]
   * */
  fitTransform(data: ScikitVecOrMatrix) {
    this.fit(data)
    return this.transform(data)
  }

  /**
   * Inverse transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new StandardScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.transform([1, 2, 3, 4, 5])
   * // [0.0, 0.0, 0.0, 0.0, 0.0]
   * scaler.inverseTransform([0.0, 0.0, 0.0, 0.0, 0.0])
   * // [1, 2, 3, 4, 5]
   * */
  inverseTransform(data: ScikitVecOrMatrix) {
    const tensorArray = convertToNumericTensor1D_2D(data)
    const outputData = tensorArray.mul(this.$std).add(this.$mean)

    return convertTensorToInputType(outputData, data)
  }
}
