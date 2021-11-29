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

import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
import { Scikit1D, Scikit2D } from '../types'
import { assert, isScikit1D, isScikit2D } from '../types.utils'
import { median, quantileSeq } from 'mathjs'
import { tensor1d, Tensor1D } from '@tensorflow/tfjs-core'
import { RegressorMixin } from '../mixins'

/*
Next steps:
0. Implement score method
1. Make the y variable in fit method work against 1D or 2D objects
2. Run against all tests in scikit-learn
*/

export interface DummyRegressorParams {
  /**
   * The strategy that this DummyRegressor will use to make a prediction.
   * Accepted values are 'mean', 'median', 'constant', and 'quantile'.
   *
   * If 'mean' is chosen then the DummyRegressor will just return the 'mean'
   * of the target variable as it's prediction.
   *
   * Likewise with 'median'.
   *
   * If "constant" is chosen, you will have to supply the constant number, and this regressor will always
   * return that value.
   *
   * If "quantile" is chosen, you'll have to chosen the quantile value between 0 < `quantile` < 1.
   * And that value will be returned always. **default = mean**
   */
  strategy?: 'mean' | 'median' | 'constant' | 'quantile'

  /**
   * In the case where you chose 'constant' as your strategy, this number
   * will be the number that is predicted for any input.
   *
   * Every constructor parameter is used as a class variable as well.
   * If "mean", "median", or "quantile" are chosen the class variable "constant" will be
   * set with the "mean", "median", or "quantile" after fit.
   */
  constant?: number

  /**
   * The quantile to predict in the quantile strategy.
   * 0.5 is the median. 0.0 is the min. 1.0 is the max
   */
  quantile?: number
}

/** Builds a regressor with simple rules.
 *
 * @example
 * ```js
 * import { DummyRegressor } from 'scikitjs'
 * const reg = new DummyRegressor({ strategy: 'mean' })

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]
    const y = [10, 20, 30] // The mean is 20
    reg.fit(X, y) // This regressor will return 20 for any input
 * ```
 */
export class DummyRegressor extends RegressorMixin {
  strategy: string
  constant?: number
  quantile?: number

  constructor({
    strategy = 'mean',
    constant,
    quantile
  }: DummyRegressorParams = {}) {
    super()
    this.strategy = strategy
    this.constant = constant
    this.quantile = quantile
  }

  public fit(X: Scikit2D, y: Scikit1D): DummyRegressor {
    assert(isScikit1D(y), 'y variable can not be converted to a 1D Tensor.')
    assert(
      ['mean', 'median', 'constant', 'quantile'].includes(this.strategy),
      `Strategy ${this.strategy} not supported. We support 'mean', 'median', 'constant', and 'quantile'`
    )

    const newY = convertToNumericTensor1D(y)

    if (this.strategy === 'mean') {
      this.constant = newY.mean().dataSync()[0]
      return this
    }
    if (this.strategy === 'median') {
      this.constant = median(newY.arraySync() as number[])
      return this
    }
    if (this.strategy === 'quantile') {
      assert(
        typeof this.quantile === 'number' &&
          !isNaN(this.quantile) &&
          isFinite(this.quantile),
        'quantile is not set to a number. Please set it to a value between 0 and 1 in the constructor'
      )
      assert(
        (this.quantile as number) < 0 || (this.quantile as number) > 1,
        'quantile must be set to a value between 0 and 1'
      )
      this.constant = quantileSeq(
        newY.arraySync() as number[],
        this.quantile as number
      ) as number
      return this
    }
    // Handles 'constant' case
    return this
  }

  public predict(X: Scikit2D): Tensor1D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    let newData = convertToNumericTensor2D(X)
    let length = newData.shape[0]
    return tensor1d(Array(length).fill(this.constant))
  }
}
