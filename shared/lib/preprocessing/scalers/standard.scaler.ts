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
import { Scikit2D } from '../../types'
import { isScikit2D, assert } from '../../types.utils'
import { tensorMean, tensorStd, turnZerosToOnes } from '../../math'
import { TransformerMixin } from '../../mixins'

import { tf } from '../../../globals'

/*
Next steps:
1. Implement withMean, and withStd
2. Test on the scikit-learn tests
*/

/**
 * Standardize features by removing the mean and scaling to unit variance.
 * The standard score of a sample x is calculated as: $z = (x - u) / s$,
 * where $u$ is the mean of the training samples, and $s$ is the standard deviation of the training samples.
 *
 * @example
 * ```js
 * import { StandardScaler } from 'scikitjs'
 *
 * const data = [
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1]
    ]

    const scaler = new StandardScaler()
    const expected = scaler.fitTransform(data)
    // const expected = [
    //  [-1, -1],
    //  [-1, -1],
    //  [1, 1],
    //  [1, 1]
    // ]
 * ```
 */
export class StandardScaler extends TransformerMixin {
  /** The per-feature scale that we see in the dataset. We divide by this number. */
  scale: tf.Tensor

  /** The per-feature mean that we see in the dataset. We subtract by this number. */
  mean: tf.Tensor

  constructor() {
    super()
    this.scale = tf.tensor1d([])
    this.mean = tf.tensor1d([])
  }

  /**
   * Fit a StandardScaler to the data.
   */
  public fit(X: Scikit2D): StandardScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const std = tensorStd(tensorArray, 0, true)
    this.mean = tensorMean(tensorArray, 0, true)

    // Deal with zero variance issues
    this.scale = turnZerosToOnes(std) as tf.Tensor1D
    return this
  }

  /**
   * Transform the data using the fitted scaler
   */
  public transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.sub(this.mean).div<tf.Tensor2D>(this.scale)
    return outputData
  }

  /**
   * Inverse transform the data using the fitted scaler
   */
  public inverseTransform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.mul(this.scale).add<tf.Tensor2D>(this.mean)
    return outputData
  }
}
