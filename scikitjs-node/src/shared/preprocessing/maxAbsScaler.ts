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
import { assert, isScikit2D } from '../typesUtils'
import { tensorMax, turnZerosToOnes } from '../math'
import { TransformerMixin } from '../mixins'
import { Scikit2D } from '../types'
import { tf, dfd } from '../../globals'

/*
Next steps:
0. Write the maxabsScale function (takes in 1D and 2D arrays)
1. Support maxAbs property on object
2. Support streaming with partialFit
3. getFeatureNamesOut
*/

/** MaxAbsScaler scales the data by dividing by the max absolute value that it finds per feature.
 * It's a useful scaling if you wish to keep sparsity in your dataset.
 *
 * @example
 * ```js
 * import { MaxAbsScaler } from 'scikitjs'
 *
 * const scaler = new MaxAbsScaler()
   const data = [
     [-1, 5],
     [-0.5, 5],
     [0, 10],
     [1, 10]
   ]

   const expected = scaler.fitTransform(data)
   const above = [
    [-1, 0.5],
    [-0.5, 0.5],
    [0, 1],
    [1, 1]
   ]
 *
 * ```
*/
export class MaxAbsScaler extends TransformerMixin {
  /** The per-feature scale that we see in the dataset. We divide by this number. */
  scale: tf.Tensor1D

  /** The number of features seen during fit */
  nFeaturesIn: number

  /** The number of samples processed by the Estimator. Will be reset on new calls to fit */
  nSamplesSeen: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'maxabsscaler'

  constructor() {
    super()
    this.scale = tf.tensor1d([])
    this.nFeaturesIn = 0
    this.nSamplesSeen = 0
    this.featureNamesIn = []
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
