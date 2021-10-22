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

import {
  convertToNumericTensor1D,
  convertToNumericTensor1D_2D
} from '../utils'
import { Scikit1D, ScikitVecOrMatrix } from '../types'
import { isScikitVecOrMatrix, assert, isScikit1D } from '../types.utils'
import { median } from 'simple-statistics'
import { PredictorMixin } from '../mixins'

/**
 * Supported strategies for DummyRegressor
 */

type Strategy = 'mean' | 'median' | 'constant'

/**
 * Creates an estimator that guesses a prediction based on simple rules.
 */
export default class DummyRegressor extends PredictorMixin {
  $fill: number
  $strategy: string

  constructor(strategy: Strategy = 'mean', fill?: number) {
    super()
    this.$fill = fill || 0
    this.$strategy = strategy
  }

  /**
   * Fit a DummyClassifier to the data.
   * @param X Array, Tensor, DataFrame or Series object
   * @param y Array, Series object
   * @returns DummyClassifier
   * @example
   * const dummy = new DummyClassifier()
   * dummy.fit([[1,1], [2,2], [3,3]],[1, 2, 3])
   */
  fit(X: ScikitVecOrMatrix, y: Scikit1D): DummyRegressor {
    assert(isScikit1D(y), 'Data can not be converted to a 1D or 2D matrix.')
    assert(
      ['mean', 'median', 'constant'].includes(this.$strategy),
      `Strategy ${this.$strategy} not supported. We support 'mean', 'median', and 'constant'`
    )

    const newY = convertToNumericTensor1D(y)

    if (this.$strategy === 'mean') {
      this.$fill = newY.mean().dataSync()[0]
      return this
    }
    if (this.$strategy === 'median') {
      this.$fill = median(newY.arraySync() as number[])
      return this
    }
    // Handles 'constant' case
    return this
  }

  /**
   * Predicts response on given example data
   * @param X Array, Tensor, DataFrame or Series object
   * @returns Array, Tensor, DataFrame or Series object
   * @example
   * const dummy = new DummyRegressor('median')
   * dummy.fit([1, 3, 3, 10, 20])
   * dummy.predict([1, 2, 3, 4, 5])
   * // [3, 3, 3, 3, 3]
   * */
  predict(X: ScikitVecOrMatrix) {
    assert(
      isScikitVecOrMatrix(X),
      'Data can not be converted to a 1D or 2D matrix.'
    )
    let newData = convertToNumericTensor1D_2D(X)
    let length = newData.shape[0]
    return Array(length).fill(this.$fill)
  }
}
