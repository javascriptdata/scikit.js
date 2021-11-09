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
import { Tensor, Tensor1D, tensor1d } from '@tensorflow/tfjs'
import { Series } from 'danfojs'
import { is1DArray } from '../../utils'
import { TransformerMixin } from '../../mixins'
import { Scikit1D } from '../../types'

/**
 * Encode target labels with value between 0 and n_classes-1.
 */
export default class LabelEncoder extends TransformerMixin {
  private $labels: Map<string | number | boolean, number>

  constructor() {
    super()
    this.$labels = new Map<string | number | boolean, number>()
  }

  convertTo1DArray(X: Scikit1D): Iterable<string | number | boolean> {
    if (X instanceof Series) {
      return X.values as any[]
    }
    if (X instanceof Tensor) {
      return X.arraySync()
    }
    return X
  }

  /**
   * Maps values to unique integer labels between 0 and n_classes-1.
   * @param data 1d array of labels, Tensor, or  Series to fit.
   * @example
   * ```
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * ```
   */
  fit(X: Scikit1D): LabelEncoder {
    const arr = this.convertTo1DArray(X)
    const dataSet = Array.from(new Set(arr))

    dataSet.forEach((value, index) => {
      this.$labels.set(value, index)
    })
    return this
  }

  /**
   * Encode labels with value between 0 and n_classes-1.
   * @param data 1d array of labels, Tensor, or  Series to be encoded.
   * @example
   * ```
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * console.log(encoder.transform(["a", "b", "c", "d"]))
   * // [0, 1, 2, 3]
   * ```
   */
  transform(X: Scikit1D): Tensor1D {
    const arr = this.convertTo1DArray(X)
    const encodedData = (arr as any).map((value: any) => {
      let val = this.$labels.get(value)
      return val === undefined ? -1 : val
    })
    return tensor1d(encodedData)
  }

  /**
   * Inverse transform values back to original values.
   * @param data 1d array of labels, Tensor, or  Series to be decoded.
   * @example
   * ```
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * console.log(encoder.inverseTransform([0, 1, 2, 3]))
   * // ["a", "b", "c", "d"]
   * ```
   */
  inverseTransform(X: Scikit1D): any[] {
    const arr = this.convertTo1DArray(X)
    const invMap = new Map(Array.from(this.$labels, (a) => a.reverse()) as any)

    const tempData = (arr as any).map((value: any) => {
      return invMap.get(value) === undefined ? null : invMap.get(value)
    })
    return tempData
  }

  /**
   * Get the number of classes.
   * @returns number of classes.
   * @example
   * ```
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * console.log(encoder.nClasses)
   * // 4
   * ```
   */
  get nClasses(): number {
    return this.$labels.size
  }

  /**
   * Get the mapping of classes to integers.
   * @returns mapping of classes to integers.
   * @example
   * ```
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * console.log(encoder.classes)
   * // {a: 0, b: 1, c: 2, d: 3}
   * ```
   */
  get classes(): Map<string | number | boolean, number> {
    return this.$labels
  }
}
