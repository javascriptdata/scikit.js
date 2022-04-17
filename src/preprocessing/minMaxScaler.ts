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
import { Scikit2D, Transformer } from '../types'
import { isScikit2D, assert, isDataFrameInterface } from '../typesUtils'
import { tensorMin, tensorMax, turnZerosToOnes } from '../math'
import { TransformerMixin } from '../mixins'
import { tf } from '../shared/globals'

/*
Next steps:
1. Implement constructor arg "clip"
2. partialFit for online scaling
3. getFeatureNamesOut
4. Pass next 5 scikit-learn tests
*/

export interface MinMaxScalerParams {
  /** Desired range of transformed data. **default = [0, 1] ** */
  featureRange?: [number, number]
}

/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 *
 * @example
 * ```js
 * import { MinMaxScaler } from 'scikitjs'
 *
 * const data = [
      [-1, 2],
      [-0.5, 6],
      [0, 10],
      [1, 18]
    ]
    const scaler = new MinMaxScaler()
    const expected = scaler.fitTransform(data)
    // const expected = [
    //  [0, 0],
    //  [0.25, 0.25],
    //  [0.5, 0.5],
    //  [1, 1]
    //]
    ```
 */

export class MinMaxScaler extends TransformerMixin implements Transformer {
  featureRange: [number, number]

  /** The per-feature scale that we see in the dataset. */
  scale: tf.Tensor1D

  min: tf.Tensor1D

  /** The per-feature minimum that we see in the dataset. */
  dataMin: tf.Tensor1D
  /** The per-feature maximum that we see in the dataset. */
  dataMax: tf.Tensor1D
  /** The per-feature range that we see in the dataset. */
  dataRange: tf.Tensor1D
  /** The number of features seen during fit */
  nFeaturesIn: number

  /** The number of samples processed by the Estimator. Will be reset on new calls to fit */
  nSamplesSeen: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'minmaxscaler'

  constructor({ featureRange = [0, 1] }: MinMaxScalerParams = {}) {
    super()
    this.featureRange = featureRange

    this.scale = tf.tensor1d([])
    this.min = tf.tensor1d([])
    this.dataMin = tf.tensor1d([])
    this.dataMax = tf.tensor1d([])
    this.dataRange = tf.tensor1d([])

    this.nFeaturesIn = 0
    this.nSamplesSeen = 0
    this.featureNamesIn = []
  }

  isNumber(value: any) {
    return typeof value === 'number' && isFinite(value)
  }
  /**
   * Fits a MinMaxScaler to the data
   */
  public fit(X: Scikit2D): MinMaxScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    assert(
      this.isNumber(this.featureRange[0]) &&
        this.isNumber(this.featureRange[1]) &&
        this.featureRange[0] < this.featureRange[1],
      'featureRange needs to contain exactly two numbers where the first is less than the second'
    )
    const tensorArray = convertToNumericTensor2D(X)
    const max = tensorMax(tensorArray, 0, true) as tf.Tensor1D
    const min = tensorMin(tensorArray, 0, true) as tf.Tensor1D
    const range = max.sub<tf.Tensor1D>(min)
    this.scale = tf.div(
      this.featureRange[1] - this.featureRange[0],
      turnZerosToOnes(range)
    )
    this.min = tf.sub(this.featureRange[0], min.mul(this.scale))
    this.dataMin = min
    this.dataMax = max
    this.dataRange = range
    this.nSamplesSeen = tensorArray.shape[0]
    this.nFeaturesIn = tensorArray.shape[1]
    if (isDataFrameInterface(X)) {
      this.featureNamesIn = [...X.columns]
    }
    return this
  }

  /**
   * Transform the data using the fitted scaler
   * */
  public transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.mul(this.scale).add<tf.Tensor2D>(this.min)
    return outputData
  }

  /**
   * Inverse transform the data using the fitted scaler
   * */
  public inverseTransform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    const outputData = tensorArray.sub(this.min).div<tf.Tensor2D>(this.scale)
    return outputData
  }
}
