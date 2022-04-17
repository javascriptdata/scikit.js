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
0. Support inverseTransform
1. Maybe support dtype constructor arg
2. Shouldn't OrdinalEncoder support partialFit, seems like that might be useful
3. Pass the next 5 tests
*/

export interface OrdinalEncoderParams {
  /**
   * Categories (unique values) per feature:
   * ‘auto’ : Determine categories automatically from the training data.
   * list : categories[i] holds the categories expected in the ith column.
   * The passed categories should not mix strings and numeric values, and should be sorted in case of numeric values.
   * **default = "auto"**
   */
  categories?: 'auto' | (number | string | boolean)[][]

  /** When set to ‘error’ an error will be raised in case an unknown categorical
   * feature is present during transform. When set to ‘use_encoded_value’,
   * the encoded value of unknown categories will be set to the value
   * given for the parameter unknown_value.
   * In inverse_transform, an unknown category will be denoted as null.
   * **default = "error"**
   */
  handleUnknown?: 'error' | 'useEncodedValue'

  /**When the parameter handle_unknown is set to ‘use_encoded_value’, this parameter
   * is required and will set the encoded value of unknown categories.
   * It has to be distinct from the values used to encode any of the categories in fit.
   * Great choices for this number are NaN or -1. **default = NaN** */
  unknownValue?: number
}

/**
 * Encode categorical features as an integer array.
 * The input to this transformer should be an array-like of integers or strings,
 * which represent categorical (discrete) features. The features are then converted to ordinal integers.

* @example
 * ```js
 *  const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OrdinalEncoder()
    encode.fitTransform(X) // returns the expected object below
    const expected = [
      [0, 0],
      [1, 1],
      [0, 2]
    ]
 * ```
 */
export class OrdinalEncoder extends TransformerMixin {
  categories: (number | string | boolean)[][]
  handleUnknown?: 'error' | 'useEncodedValue'
  unknownValue?: number
  /** This holds the categories parameter that is passed in the constructor. `this.categories`
   * holds the actual learned categories or the ones passed in from the constructor */
  categoriesParam: 'auto' | (number | string | boolean)[][]

  /** The number of features seen during fit */
  nFeaturesIn: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'ordinalencoder'

  constructor({
    categories = 'auto',
    handleUnknown = 'error',
    unknownValue = NaN
  }: OrdinalEncoderParams = {}) {
    super()
    this.categoriesParam = categories
    this.categories = []
    this.handleUnknown = handleUnknown
    this.unknownValue = unknownValue
    this.nFeaturesIn = 0
    this.featureNamesIn = []
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
   * Fits a OrdinalEncoder to the data.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public fit(X: Scikit2D, y?: Scikit1D): OrdinalEncoder {
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
            val = this.unknownValue
          }
        }
        curArray.push(val)
      }
      finalArray.push(curArray)
    }
    return finalArray
  }
  /**
   * Encodes the data using the fitted OrdinalEncoder.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public transform(X: Scikit2D, y?: Scikit1D): tf.Tensor2D {
    const array2D = convertScikit2DToArray(X)
    const result2D = this.loopOver2DArrayToUseLabels(array2D)
    return tf.tensor2d(result2D as number[][], undefined, 'int32')
  }
}
