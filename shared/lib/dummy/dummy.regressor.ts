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
import { PredictorMixin } from '../mixins'

/*
TODO
1. Make the y variable in fit method work against 1D or 2D objects
1. Run against all tests in scikit-learn
2. Finish docs so they are pretty
*/

export interface DummyRegressorParams {
  /**
   * The strategy that this DummyRegressor will use to make a prediction.
   * Accepted values are 'mean', 'median', and 'constant'
   *
   * If 'mean' is chosen then the DummyRegressor will just "guess" the 'mean'
   * of the target variable as it's prediction.
   */
  strategy?: 'mean' | 'median' | 'constant' | 'quantile'

  /**
   * In the case where you chose 'constant' as your strategy, the fill number
   * will be the number that is predicted for any input.
   */
  constant?: number

  /**
   * The quantile to predict in the quantile strategy.
   * 0.5 is the median. 0.0 is the min. 1.0 is the max
   */
  quantile?: number
}

export default class DummyRegressor extends PredictorMixin {
  strategy: string
  constant: number | undefined
  quantile: number | undefined

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

  /**
   * Fit a DummyClassifier to the data.
   * @param X Array, Tensor, DataFrame or Series object
   * @param y Array, Series object
   * @returns DummyClassifier
   * @example
   * const dummy = new DummyClassifier()
   * dummy.fit([[1,1], [2,2], [3,3]],[1, 2, 3])
   */
  fit(X: Scikit2D, y: Scikit1D): DummyRegressor {
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
  predict(X: Scikit2D) {
    assert(isScikit2D(X), 'Data can not be converted to a 1D or 2D matrix.')
    let newData = convertToNumericTensor2D(X)
    let length = newData.shape[0]
    return Array(length).fill(this.constant)
  }
}
