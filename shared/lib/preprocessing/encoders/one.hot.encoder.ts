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

import { convertTo2DArray } from '../../utils'
import { Scikit1D, Scikit2D } from '../../types'
import { TransformerMixin } from '../../mixins'

import { tf } from '../../../globals'

/*
Todo:
1. Implement inverseTransform for 2D array
2. Pass the next 5 scikit-learn tests
*/

/**
 * Fits a OneHotEncoder to the data.
 * @example
 * ```js
 * const encoder = new OneHotEncoder()
 * encoder.fit(["a", "b", "c"])
 * ```
 */
export default class OneHotEncoder extends TransformerMixin {
  categories: (number | string | boolean)[][]
  constructor() {
    super()
    this.categories = []
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
      // let newMap = new Map<string | number | boolean, number>()
      // results.forEach((el, i) => {
      //   newMap.set(el as number, i)
      // })
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
  fit(X: Scikit2D, y?: Scikit1D): OneHotEncoder {
    const array2D = convertTo2DArray(X)
    this.loopOver2DArrayToSetLabels(array2D)
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
        let actualIndex = val === undefined ? -1 : val
        curArray.push(actualIndex)
      }
      finalArray.push(curArray)
    }
    return finalArray
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
  transform(X: Scikit2D, y?: Scikit1D): tf.Tensor2D {
    const array2D = convertTo2DArray(X)
    const result2D = this.loopOver2DArrayToUseLabels(array2D)
    const newTensor = tf.tensor2d(result2D, undefined, 'int32')
    return tf.concat(
      newTensor
        .unstack(1)
        .map((el, i) => tf.oneHot(el, this.categories[i].length)),
      1
    ) as tf.Tensor2D
  }
  // Only works for single column OneHotEncoding
  inverseTransform(X: tf.Tensor2D): any[] {
    let labels = this.classesToMapping(this.categories[0])
    const tensorLabels = X.argMax(1) as tf.Tensor1D
    const invMap = new Map(Array.from(labels, (a) => a.reverse()) as any)

    const tempData = tensorLabels.arraySync().map((value) => {
      return invMap.get(value) === undefined ? null : invMap.get(value)
    })
    return tempData
  }
}
