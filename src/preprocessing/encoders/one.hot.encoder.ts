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

import { oneHot, tensor1d, Tensor2D, unique } from '@tensorflow/tfjs-node'
import { DataFrame, Series } from 'danfojs-node'
import { convertToTensor1D } from '../../utils'
import { Scikit1D, Scikit2D } from '../../types'
/**
 * Fits a OneHotEncoder to the data.
 * @example
 * ```js
 * const encoder = new OneHotEncoder()
 * encoder.fit(["a", "b", "c"])
 * ```
 */
export default class OneHotEncoder {
  private $labels: Map<string | number | boolean, number>

  constructor() {
    this.$labels = new Map()
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
  public fit(data: Scikit1D) {
    // I convert to a 1D Tensor first because it does all
    // the error checking to ensure that this is actually a 1D array / Tensor
    const data1D = convertToTensor1D(data)

    // I would've liked to use the tf.unique function but it's not supported in
    // node yet. Apparently it is supported in wasm, webgl though :/
    const values = Array.from(new Set(data1D.arraySync()))
    values.forEach((el, i) => {
      this.$labels.set(el, i)
    })
    return this
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
  public transform(data: Scikit1D): Scikit2D {
    const data1D = convertToTensor1D(data)
    const dataOneHotIndices = tensor1d(
      data1D.arraySync().map((el) => {
        let val = this.$labels.get(el)
        return val === undefined ? -1 : val
      }),
      'int32'
    )
    const oneHotArr = oneHot(dataOneHotIndices, this.$labels.size)

    if (data instanceof Array) {
      return oneHotArr.arraySync() as number[][]
    } else if (data instanceof Series) {
      return new DataFrame(oneHotArr, {
        index: data.index,
      })
    } else {
      return oneHotArr as Tensor2D
    }
  }

  /**
   * Fit and transform the data using the fitted OneHotEncoder.
   * @param data 1d array of labels, Tensor, or  Series to be encoded.
   * @example
   * ```js
   * const encoder = new OneHotEncoder()
   * encoder.fitTransform(["a", "b", "c"])
   * ```
   */
  public fitTransform(data: Scikit1D): Scikit2D {
    this.fit(data)
    return this.transform(data)
  }
}
