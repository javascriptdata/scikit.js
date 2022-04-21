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

import { convertScikit2DToArray } from '../utils'
import { Scikit1D, Scikit2D } from '../types'
import { TransformerMixin } from '../mixins'
import { tf } from '../shared/globals'
import { isDataFrameInterface } from '../typesUtils'

/*
Next steps:
0. Implement drop constructor option if-binary, and array
1. Implement inverseTransform for 2D array
2. getFeatureNamesOut
3. Pass the next 5 scikit-learn tests
*/

export interface OneHotEncoderParams {
  /**
   * Categories (unique values) per feature:
   * ‘auto’ : Determine categories automatically from the training data.
   * list : categories[i] holds the categories expected in the ith column.
   * The passed categories should not mix strings and numeric values, and should be sorted in case of numeric values.
   * **default = "auto"**
   */
  categories?: 'auto' | (number | string | boolean)[][]

  /** When set to ‘error’ an error will be raised in case an unknown categorical
   * feature is present during transform. When set to ‘ignore’,
   * the encoded value of will be all zeros
   * In inverse_transform, an unknown category will be denoted as null.
   * **default = "error"**
   */
  handleUnknown?: 'error' | 'ignore'

  /**
   * Specifies a methodology to use to drop one of the categories per feature.
   * This is useful in situations where perfectly collinear features cause problems, such as when
   * feeding the resulting data into a neural network or an unregularized regression.
   * However, dropping one category breaks the symmetry of the original representation and can therefore induce a bias in
   * downstream models, for instance for penalized linear classification or regression models.
   *
   * Options:
   * undefined : retain all features (the default).
   * ‘first’ : drop the first category in each feature. If only one category is present, the feature will be dropped entirely.
   * **default = undefined**
   */
  drop?: 'first'
}
/**
 * Fits a OneHotEncoder to the data.
 *
 * @example
 * ```js
 * import { OneHotEncoder } from 'scikitjs'
 *
 *
 * const X = [
    ['Male', 1],
    ['Female', 2],
    ['Male', 4]
   ]
   const encode = new OneHotEncoder()
   encode.fitTransform(X) // returns the object below
   const expected = [
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
   ]
 * ```
 */
export class OneHotEncoder extends TransformerMixin {
  /** categories is a list of unique labels per feature */
  categories: (number | string | boolean)[][]
  handleUnknown?: 'error' | 'ignore'
  /** This holds the categories parameter that is passed in the constructor. `this.categories`
   * holds the actual learned categories or the ones passed in from the constructor */
  categoriesParam: 'auto' | (number | string | boolean)[][]

  drop?: 'first'

  /** The number of features seen during fit */
  nFeaturesIn: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'OneHotEncoder'

  constructor({
    categories = 'auto',
    handleUnknown = 'error',
    drop
  }: OneHotEncoderParams = {}) {
    super()
    this.categoriesParam = categories
    this.categories = []
    this.handleUnknown = handleUnknown
    this.nFeaturesIn = 0
    this.featureNamesIn = []
    this.drop = drop
  }

  classesToMapping(
    classes: Array<string | number | boolean>
  ): Map<string | number | boolean, number> {
    const labels = new Map<string | number | boolean, number>()
    classes.forEach((value, index) => {
      labels.set(value, index)
    })
    return labels
  }

  loopOver2DArrayToSetLabels(array2D: any) {
    for (let j = 0; j < array2D[0].length; j++) {
      let curSet = new Set()
      for (let i = 0; i < array2D.length; i++) {
        curSet.add(array2D[i][j])
      }
      let results = Array.from(curSet)
      this.categories.push(results as number[])
    }
  }

  /**
   * Fits a OneHotEncoder to the data.
   * @param data 1d array of labels, Tensor, or  Series to be encoded.
   * @returns OneHotEncoder
   * @example
   * ```js
   * const encoder = new OneHotEncoder()
   * encoder.fit(["a", "b", "c"])
   * ```
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public fit(X: Scikit2D, y?: Scikit1D): OneHotEncoder {
    const array2D = convertScikit2DToArray(X)

    if (this.categoriesParam === 'auto') {
      this.loopOver2DArrayToSetLabels(array2D)
      return this
    }
    this.categories = this.categoriesParam
    this.nFeaturesIn = array2D.length === 0 ? 0 : array2D[0].length || 0
    if (isDataFrameInterface(X)) {
      this.featureNamesIn = [...X.columns]
    }

    return this
  }

  loopOver2DArrayToUseLabels(array2D: any) {
    let labels = this.categories.map((el) => this.classesToMapping(el))
    let finalArray = []
    for (let i = 0; i < array2D.length; i++) {
      let curArray = []
      for (let j = 0; j < array2D[0].length; j++) {
        let curElem = array2D[i][j]
        let val = labels[j].get(curElem)
        if (val === undefined) {
          if (this.handleUnknown === 'error') {
            throw new Error(
              `Unknown value ${curElem} encountered while transforming. Not encountered in training data`
            )
          } else {
            val = -1 // When we one hot encode this it will come back as all zeros
          }
        }
        if (this.drop === 'first') {
          val -= 1
        }
        curArray.push(val)
      }
      finalArray.push(curArray)
    }
    return finalArray
  }

  /** Generalization of the tf.oneHot that can handle "one-hotting" with a single column
   * output.
   */
  convertToOneHot(
    tensor: tf.Tensor1D,
    numberOfOneHotColumns: number
  ): tf.Tensor2D {
    if (numberOfOneHotColumns >= 2) {
      return tf.oneHot(tensor, numberOfOneHotColumns) as tf.Tensor2D
    }
    if (numberOfOneHotColumns === 1) {
      // Every integer that isn't 0 becomes 0
      tensor = tf.where(
        tensor.equal(0),
        tf.ones(tensor.shape, 'int32'),
        tf.zeros(tensor.shape, 'int32')
      )

      return tensor.reshape([-1, 1])
    }

    // Case where numberOfOneHotColumns = 0
    return tf.tensor2d([])
  }

  /**
   * Encodes the data using the fitted OneHotEncoder.
   * @param data 1d array of labels, Tensor, or  Series to be encoded.
   * @example
   * ```js
   * const encoder = new OneHotEncoder()
   * encoder.fit(["a", "b", "c"])
   * encoder.transform(["a", "b", "c"])
   * ```
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public transform(X: Scikit2D, y?: Scikit1D): tf.Tensor2D {
    const array2D = convertScikit2DToArray(X)
    const result2D = this.loopOver2DArrayToUseLabels(array2D)
    const newTensor = tf.tensor2d(result2D as number[][], undefined, 'int32')
    return tf.concat(
      newTensor.unstack<tf.Tensor1D>(1).map((el, i) => {
        let categoryNumber = this.categories[i].length
        let numberOfOneHotColumns =
          this.drop === 'first' ? categoryNumber - 1 : categoryNumber
        let val = this.convertToOneHot(el, numberOfOneHotColumns)
        return val
      }),
      1
    ) as tf.Tensor2D
  }

  /** Only works for single column OneHotEncoding */
  public inverseTransform(X: tf.Tensor2D): any[] {
    let labels = this.classesToMapping(this.categories[0])
    const tensorLabels = X.argMax(1) as tf.Tensor1D
    const invMap = new Map(Array.from(labels, (a) => a.reverse()) as any)

    const tempData = tensorLabels.arraySync().map((value) => {
      return invMap.get(value) === undefined ? null : invMap.get(value)
    })
    return tempData
  }
}
