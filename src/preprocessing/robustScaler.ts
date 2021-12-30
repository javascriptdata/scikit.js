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
import { isScikit2D, assert } from '../typesUtils'
import { turnZerosToOnes } from '../math'
import { TransformerMixin } from '../mixins'
import { quantileSeq } from 'mathjs'
import { tf, dfd } from '../shared/globals'

/*
Next steps:
1. Implement unitVariance constructor arg
2. getFeatureNamesOut
3. Test on the next 5 scikit-learn tests
*/

export interface RobustScalerParams {
  /**Quantile range used to calculate scale_. By default this is equal to the IQR, i.e.,
   * q_min is the first quantile and q_max is the third quantile.
   * Numbers must be between 0, and 100. **default [25.0, 75.0]** */
  quantileRange?: [number, number]

  /** Whether or not we should scale the data. **default = true** */
  withScaling?: boolean

  /** Whether or not we should center the data. **default = true** */
  withCentering?: boolean
}

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

/**
 * Scales the data but is robust to outliers. While StandardScaler will subtract the mean, and
 * divide by the variance, both of those measures are not robust to outliers. So instead of the mean
 * we use the median, and instead of the variance we use the Interquartile Range (which is the distance
 * between the quantile .25, and quantile .75).
 *
 * @example
 * ```js
 * import { RobustScaler } from 'scikitjs'
 *
    const X = [
      [1, -2, 2],
      [-2, 1, 3],
      [4, 1, -2]
    ]

    const scaler = new RobustScaler()
    scaler.fitTransform(X)

    const result = [
      [0, -2, 0],
      [-1, 0, 0.4],
      [1, 0, -1.6]
    ]
 * ```
 */
export class RobustScaler extends TransformerMixin {
  /** The per-feature scale that we see in the dataset. We divide by this number. */
  scale: tf.Tensor1D

  /** The per-feature median that we see in the dataset. We subtrace this number. */
  center: tf.Tensor1D

  /** The number of features seen during fit */
  nFeaturesIn: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  quantileRange: [number, number]
  withScaling: boolean
  withCentering: boolean

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'robustscaler'

  constructor({
    quantileRange = [25.0, 75.0],
    withCentering = true,
    withScaling = true
  }: RobustScalerParams = {}) {
    super()
    this.scale = tf.tensor1d([])
    this.center = tf.tensor1d([])
    this.quantileRange = quantileRange
    this.withScaling = withScaling
    this.withCentering = withCentering
    this.nFeaturesIn = 0
    this.featureNamesIn = []
  }

  isNumber(value: any) {
    return typeof value === 'number' && isFinite(value)
  }
  public fit(X: Scikit2D): RobustScaler {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    assert(
      this.isNumber(this.quantileRange[0]) &&
        this.isNumber(this.quantileRange[1]),
      'quantileRange values must be numbers'
    )
    let lowPercentile = this.quantileRange[0]
    let highPercentile = this.quantileRange[1]
    assert(
      lowPercentile < highPercentile &&
        0 <= lowPercentile &&
        lowPercentile <= 100 &&
        0 <= highPercentile &&
        highPercentile <= 100,
      'quantileRange numbers must be between 0 and 100'
    )

    const tensorArray = convertToNumericTensor2D(X)
    const rowOrientedArray = tensorArray.transpose<tf.Tensor2D>().arraySync()

    if (this.withCentering) {
      const quantiles = rowOrientedArray.map((arr: number[] | string[]) =>
        quantileSeq(removeMissingValuesFromArray(arr), 0.5)
      )
      this.center = tf.tensor1d(quantiles as number[])
    }
    if (this.withScaling) {
      const quantiles = rowOrientedArray.map((arr: number[] | string[]) =>
        quantileSeq(removeMissingValuesFromArray(arr), [
          lowPercentile / 100,
          highPercentile / 100
        ])
      )
      const scale = tf.tensor1d(quantiles.map((el: any) => el[1] - el[0]))

      // But what happens if max = min, ie.. we are dealing with a constant vector?
      // In the case above, scale = max - min = 0 and we'll divide by 0 which is no bueno.
      // The common practice in cases where the vector is constant is to change the 0 elements
      // in scale to 1, so that the division doesn't fail. We do that below
      this.scale = turnZerosToOnes(scale) as tf.Tensor1D
    }

    this.nFeaturesIn = tensorArray.shape[1]
    if (X instanceof dfd.DataFrame) {
      this.featureNamesIn = [...X.columns]
    }
    return this
  }

  public transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    let tensorArray = convertToNumericTensor2D(X)

    if (this.withCentering) {
      tensorArray = tensorArray.sub(this.center)
    }
    if (this.withScaling) {
      tensorArray = tensorArray.div<tf.Tensor2D>(this.scale)
    }
    return tensorArray
  }

  public inverseTransform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    let tensorArray = convertToNumericTensor2D(X)

    if (this.withScaling) {
      tensorArray = tensorArray.mul<tf.Tensor2D>(this.scale)
    }
    if (this.withCentering) {
      tensorArray = tensorArray.add(this.center)
    }
    return tensorArray
  }
}
