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

import { Tensor, tensor1d } from '@tensorflow/tfjs-node'
import { DataFrame, Series } from 'danfojs-node'
import { convertToTensor } from 'utils'

/**
 * Transform features by scaling each feature to a given range.
 * This estimator scales and translates each feature individually such
 * that it is in the given range on the training set, e.g. between the maximum and minimum value.
 */

export interface possibleUserData {
  data: number[] | number[][] | Tensor | DataFrame | Series
}
export default class MinMaxScaler {
  _max: Tensor
  _min: Tensor

  constructor() {
    this._max = tensor1d([])
    this._min = tensor1d([])
  }

  /**
   * Fits a MinMaxScaler to the data
   * @param data Array, Tensor, DataFrame or Series object
   * @returns MinMaxScaler
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * // MinMaxScaler {
   * //   _max: [5],
   * //   _min: [1]
   * // }
   *
   */
  fit(data: number[] | number[][] | Tensor | DataFrame | Series) {
    const tensorArray = convertToTensor(data)
    this._max = tensorArray.max(0)
    this._min = tensorArray.min(0)
    return this
  }

  /**
   * Transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.transform([1, 2, 3, 4, 5])
   * // [0, 0.25, 0.5, 0.75, 1]
   * */
  transform(data: number[] | number[][] | Tensor | DataFrame | Series) {
    const tensorArray = convertToTensor(data)
    const outputData = tensorArray.sub(this._min).div(this._max.sub(this._min))

    if (Array.isArray(data)) {
      return outputData.arraySync()
    } else if (data instanceof Series) {
      return new Series(outputData, {
        index: data.index,
      })
    } else if (data instanceof DataFrame) {
      return new DataFrame(outputData, {
        index: data.index,
        columns: data.columns,
      })
    } else {
      return outputData
    }
  }

  /**
   * Fit the data and transform it
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fitTransform([1, 2, 3, 4, 5])
   * // [0, 0.25, 0.5, 0.75, 1]
   * */
  fitTransform(data: number[] | number[][] | Tensor | DataFrame | Series) {
    this.fit(data)
    return this.transform(data)
  }

  /**
   * Inverse transform the data using the fitted scaler
   * @param data Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const scaler = new MinMaxScaler()
   * scaler.fit([1, 2, 3, 4, 5])
   * scaler.inverseTransform([0, 0.25, 0.5, 0.75, 1])
   * // [1, 2, 3, 4, 5]
   * */
  inverseTransform(data: number[] | number[][] | Tensor | DataFrame | Series) {
    const tensorArray = convertToTensor(data)
    const outputData = tensorArray.mul(this._max.sub(this._min)).add(this._min)

    if (Array.isArray(data)) {
      return outputData.arraySync()
    } else if (data instanceof Series) {
      return new Series(outputData, {
        index: data.index,
      })
    } else if (data instanceof DataFrame) {
      return new DataFrame(outputData, {
        index: data.index,
        columns: data.columns,
      })
    } else {
      return outputData
    }
  }
}
