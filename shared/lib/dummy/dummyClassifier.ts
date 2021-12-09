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
import { isScikit2D, assert, isScikit1D } from '../types.utils'
import { modeFast } from 'simple-statistics'
import { uniq, sample } from 'lodash'
import { ClassifierMixin } from '../mixins'

/*
Next steps:
0. Don't use constant as the "fill" value. Simply add the other attributes like
n_classes, and class_prior to determine what to do in each case
1. Support strategies “stratified”, “prior”
2. Support randomState in constructor for deterministic tests of 'stratified', and 'uniform'
3. Pass next 5 tests in scikit-learn
*/

/**
 * Supported strategies for this classifier.
 */

export interface DummyClassifierParams {
  /**
   * If strategy is "mostFrequent" than the most frequent class label is chosen no matter the input.
   * If "uniform" is chosen than a uniformly random class label is chosen for a given input.
   * If "constant" is chosen than you must supply a constant number and this classifier returns that number
   * a given input. **default = "mostFrequent"**
   */
  strategy?: 'mostFrequent' | 'uniform' | 'constant'
  /**
   * If strategy is "constant" than this number is returned for every input. **default = undefined**
   */
  constant?: number
}

/**
 * Creates an classifier that guesses a class label based on simple rules.
 * By setting a strategy (ie 'mostFrequent', 'uniform', or 'constant'),
 * you can create a simple classifier which can be helpful in determining
 * if a more complicated classifier is actually more predictive.
 *
 * @example
 * ```js
 * import { DummyClassifier } from 'scikitjs'
 *
 * const clf = new DummyClassifier({ strategy: 'mostFrequent' })
    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]
    const y = [10, 20, 20] // 20 is the most frequent class label
    clf.fit(X, y) // always predicts 20

    clf.predict([
      [0, 0],
      [1000, 1000]
    ]) // [20, 20]


 * ```
 *
 */

export class DummyClassifier extends ClassifierMixin {
  constant: number
  strategy: string

  /**These are the unique class labels that are seen during fit. */
  classes: number[] | string[]

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'dummyclassifier'

  constructor({
    strategy = 'mostFrequent',
    constant = 0
  }: DummyClassifierParams = {}) {
    super()
    this.constant = constant
    this.strategy = strategy
    this.classes = []
  }

  /**
   * Fit a DummyClassifier to the data.
   */
  public fit(X: Scikit2D, y: Scikit1D): DummyClassifier {
    assert(isScikit1D(y), 'Data can not be converted to a 1D or 2D matrix.')
    assert(
      ['mostFrequent', 'uniform', 'constant'].includes(this.strategy),
      `Strategy ${this.strategy} not supported. We support 'mostFrequent', 'uniform', and 'constant'`
    )

    const newY = convertToNumericTensor1D(y)

    if (this.strategy === 'mostFrequent') {
      this.constant = modeFast(newY.arraySync())
      return this
    }
    if (this.strategy === 'uniform') {
      this.classes = uniq(newY.arraySync() as number[])
      return this
    }
    // Handles 'constant' case
    return this
  }

  public predict(X: Scikit2D) {
    assert(isScikit2D(X), 'Data can not be converted to a 1D or 2D matrix.')
    assert(
      ['mostFrequent', 'uniform', 'constant'].includes(this.strategy),
      `Strategy ${this.strategy} not supported. We support 'mostFrequent', 'uniform', and 'constant'`
    )
    let newData = convertToNumericTensor2D(X)
    let length = newData.shape[0]
    if (this.strategy === 'mostFrequent' || this.strategy === 'constant') {
      return Array(length).fill(this.constant)
    }

    // "Uniform case"
    let returnArr = []
    for (let i = 0; i < length; i++) {
      returnArr.push(sample(this.classes))
    }
    return returnArr
  }
}
