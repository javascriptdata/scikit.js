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

import { Tensor1D, tensor1d, Tensor2D } from '@tensorflow/tfjs-node'
import { convertToNumericTensor2D } from '../../utils'
import { Scikit2D } from '../../types'
import { isScikit2D, assert } from '../../types.utils'
import { turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'
import { quantileSeq } from 'mathjs'
/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isEmpty(value: any) {
  return (
    value === undefined ||
    value === null ||
    (isNaN(value) && typeof value !== 'string')
  )
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function removeMissingValuesFromArray(arr: any[]) {
  const values = arr.filter((val) => {
    return !isEmpty(val)
  })
  return values
}

export default class RobustScaler extends TransformerMixin {
  $scale: Tensor1D
  $center: Tensor1D

  constructor() {
    super()
    this.$scale = tensor1d([])
    this.$center = tensor1d([])
  }

  /**
   * Fits a RobustScaler to the data
   * @param data Array, Tensor, DataFrame or Series object
   * @returns RobustScaler
   * @example
   * const scaler = new RobustScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * // RobustScaler {
   * //   $max: [5],
   * //   $min: [1]
   * // }
   *
   */
  fit(X: Scikit2D): RobustScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')

    const tensorArray = convertToNumericTensor2D(X)
    const quantiles = tensorArray
      .transpose<Tensor2D>()
      .arraySync()
      .map((arr: number[] | string[]) =>
        quantileSeq(removeMissingValuesFromArray(arr), [0.25, 0.5, 0.75])
      )

    this.$center = tensor1d(quantiles.map((el: any) => el[1]))
    const scale = tensor1d(quantiles.map((el: any) => el[2] - el[0]))

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
   * const scaler = new RobustScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.transform([1, 2, 3, 4, 5])
   * // [0, 0.25, 0.5, 0.75, 1]
   * */
  transform(X: Scikit2D): Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.sub(this.$center).div<Tensor2D>(this.$scale)
    return outputData
  }

  /**
   * Inverse transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new RobustScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.inverseTransform([0, 0.25, 0.5, 0.75, 1])
   * // [1, 2, 3, 4, 5]
   * */
  inverseTransform(X: Scikit2D): Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.mul(this.$scale).add<Tensor2D>(this.$min)
    return outputData
  }
}
