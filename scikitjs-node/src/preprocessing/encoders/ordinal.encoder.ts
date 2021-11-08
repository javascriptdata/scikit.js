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

import { tensor2d, Tensor2D } from '@tensorflow/tfjs-node'
import { convertTo2DArray } from '../../utils'
import { Scikit1D, Scikit2D } from '../../types'
import { TransformerMixin } from '../../mixins'
/**
 * Fits a OrdinalEncoder to the data.
 * @example
 * ```js
 * const encoder = new OrdinalEncoder()
 * encoder.fit(["a", "b", "c"])
 * ```
 */
export default class OrdinalEncoder extends TransformerMixin {
  $labels: Map<string | number | boolean, number>[]

  constructor() {
    super()
    this.$labels = []
  }

  loopOver2DArrayToSetLabels(array2D: any) {
    for (let j = 0; j < array2D[0].length; j++) {
      let curSet = new Set()
      for (let i = 0; i < array2D.length; i++) {
        curSet.add(array2D[i][j])
      }
      let results = Array.from(curSet)
      let newMap = new Map<string | number | boolean, number>()
      results.forEach((el, i) => {
        newMap.set(el as number, i)
      })
      this.$labels.push(newMap)
    }
  }

  /**
   * Fits a OrdinalEncoder to the data.
   * @param data 1d array of labels, Tensor, or  Series to be encoded.
   * @returns OrdinalEncoder
   * @example
   * ```js
   * const encoder = new OrdinalEncoder()
   * encoder.fit(["a", "b", "c"])
   * ```
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  fit(X: Scikit2D, y?: Scikit1D): OrdinalEncoder {
    const array2D = convertTo2DArray(X)
    this.loopOver2DArrayToSetLabels(array2D)
    return this
  }

  loopOver2DArrayToUseLabels(array2D: any) {
    let finalArray = []
    for (let i = 0; i < array2D.length; i++) {
      let curArray = []
      for (let j = 0; j < array2D[0].length; j++) {
        let curElem = array2D[i][j]
        let val = this.$labels[j].get(curElem)
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
  transform(X: Scikit2D, y?: Scikit1D): Tensor2D {
    const array2D = convertTo2DArray(X)
    const result2D = this.loopOver2DArrayToUseLabels(array2D)
    return tensor2d(result2D, undefined, 'int32')
  }
}
