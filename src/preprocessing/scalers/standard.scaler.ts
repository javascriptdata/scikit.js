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

import { tensor1d, Tensor, Tensor1D, Tensor2D } from '@tensorflow/tfjs-node'
import { convertToNumericTensor2D } from '../../utils'
import { Scikit2D } from '../../types'
import { isScikit2D, assert } from '../../types.utils'
import { tensorMean, tensorStd, turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'

/**
 * Standardize features by removing the mean and scaling to unit variance.
 * The standard score of a sample x is calculated as: `z = (x - u) / s`,
 * where `u` is the mean of the training samples, and `s` is the standard deviation of the training samples.
 */
export default class StandardScaler extends TransformerMixin {
  $std: Tensor
  $mean: Tensor

  constructor() {
    super()
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
  fit(data: Scikit2D): StandardScaler {
    assert(isScikit2D(data), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(data)
    const std = tensorStd(tensorArray, 0, true)
    this.$mean = tensorMean(tensorArray, 0, true)

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
  transform(data: Scikit2D): Tensor2D {
    assert(isScikit2D(data), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(data)
    const outputData = tensorArray.sub(this.$mean).div<Tensor2D>(this.$std)
    return outputData
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
  inverseTransform(data: Scikit2D): Tensor2D {
    assert(isScikit2D(data), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(data)
    const outputData = tensorArray.mul(this.$std).add<Tensor2D>(this.$mean)
    return outputData
  }
}
