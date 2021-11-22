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
import { assert, isScikit2D } from '../../types.utils'
import { tensorMax, turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'
import { Scikit2D } from '../../types'

import { tf } from '../../../globals'

/*
Next steps:
1. Pass the next 5 scikit-learn tests
*/

/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 */

/** MaxAbsScaler scales the data by dividing by the max absolute value that it finds per feature.
 * It's a useful scaling if you wish to keep sparsity in your dataset.
 *
 * @example
 * ```js
 * import {MaxAbsScaler} from 'scikitjs'
 *
 * const scaler = new MaxAbsScaler()
   const data = [
     [-1, 5],
     [-0.5, 5],
     [0, 10],
     [1, 10]
   ]

   const expected = scaler.fitTransform(data)
   //  const expected = [
   //   [-1, 0.5],
   //   [-0.5, 0.5],
   //   [0, 1],
   //   [1, 1]
   // ]
 *
 * ```
*/
export class MaxAbsScaler extends TransformerMixin {
  /** The per-feature scale that we see in the dataset. We divide by this number. */
  scale: tf.Tensor1D

  constructor() {
    super()
    this.scale = tf.tensor1d([])
  }

  /**
   * Fits a MinMaxScaler to the data
   */
  public fit(X: Scikit2D): MaxAbsScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const scale = tensorMax(tensorArray.abs(), 0, true) as tf.Tensor1D

    // Deal with 0 scale values
    this.scale = turnZerosToOnes(scale) as tf.Tensor1D

    return this
  }

  /**
   * Transform the data using the fitted scaler
   */
  public transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.div<tf.Tensor2D>(this.scale)
    return outputData
  }

  /**
   * Inverse transform the data using the fitted scaler
   */
  public inverseTransform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.mul<tf.Tensor2D>(this.scale)
    return outputData
  }
}
