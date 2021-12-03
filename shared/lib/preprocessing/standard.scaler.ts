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

import { convertToNumericTensor2D } from '../utils'
import { Scikit2D } from '../types'
import { isScikit2D, assert } from '../types.utils'
import { tensorMean, tensorStd, turnZerosToOnes } from '../math'
import { TransformerMixin } from '../mixins'
import { tf, dfd } from '../../globals'

/*
Next steps:
0. Implement partialFit for online learning
1. sampleWeight for fit/partialFit calls
2. class property "var"
3. getFeatureNamesOut
4. Test on the scikit-learn tests
*/

export interface StandardScalerParams {
  /** Whether or not we should subtract the mean. **default = true** */
  withMean?: boolean

  /** Whether or not we should divide by the standard deviation. **default = true** */
  withStd?: boolean
}

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

  /** Whether or not we should subtract the mean */
  withMean: boolean

  /** Whether or not we should divide by the standard deviation */
  withStd: boolean

  /** The number of features seen during fit */
  nFeaturesIn: number

  /** The number of samples processed by the Estimator. Will be reset on new calls to fit */
  nSamplesSeen: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  constructor({ withMean = true, withStd = true }: StandardScalerParams = {}) {
    super()
    this.withMean = withMean
    this.withStd = withStd
    this.scale = tf.tensor1d([])
    this.mean = tf.tensor1d([])
    this.nFeaturesIn = 0
    this.nSamplesSeen = 0
    this.featureNamesIn = []
  }

  /**
   * Fit a StandardScaler to the data.
   */
  public fit(X: Scikit2D): StandardScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    if (this.withMean) {
      this.mean = tensorMean(tensorArray, 0, true)
    }
    if (this.withStd) {
      const std = tensorStd(tensorArray, 0, true)
      // Deal with zero variance issues
      this.scale = turnZerosToOnes(std) as tf.Tensor1D
    }

    this.nSamplesSeen = tensorArray.shape[0]
    this.nFeaturesIn = tensorArray.shape[1]
    if (X instanceof dfd.DataFrame) {
      this.featureNamesIn = [...X.columns]
    }
    return this
  }

  /**
   * Transform the data using the fitted scaler
   */
  public transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    let tensorArray = convertToNumericTensor2D(X)
    if (this.withMean) {
      tensorArray = tensorArray.sub(this.mean)
    }
    if (this.withStd) {
      tensorArray = tensorArray.div(this.scale)
    }
    return tensorArray
  }

  /**
   * Inverse transform the data using the fitted scaler
   */
  public inverseTransform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    let tensorArray = convertToNumericTensor2D(X)
    if (this.withStd) {
      tensorArray = tensorArray.mul(this.scale)
    }
    if (this.withMean) {
      tensorArray = tensorArray.add(this.mean)
    }

    return tensorArray
  }
}
