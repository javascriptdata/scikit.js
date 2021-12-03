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

import { convertToNumericTensor2D } from '../utils'
import { Scikit2D } from '../types'
import { isScikit2D, assert } from '../types.utils'
import { TransformerMixin } from '../mixins'
import { tf, dfd } from '../../globals'

/*
Next steps:
1. Pass the next five scikit-learn tests
*/

export interface NormalizerParams {
  /** What kind of norm we wish to scale by. **default = "l2" ** */
  norm?: 'l2' | 'l1' | 'max'
}

/**
 * A Normalizer scales each *sample* by the $l_1$, $l_2$ or $max$ value in that sample.
 * If you imagine the input matrix as a 2D grid, then this is effectively a "horizontal" scaling (per-sample scaling)
 * as opposed to a StandardScaler which is a "vertical" scaling (per-feature scaling).
 *
 * The only input is what kind of norm you wish to scale by.
 *
 * @example
 * ```js
 * import { Normalizer } from 'scikitjs'
 *
 * const data = [
      [-1, 1],
      [-6, 6],
      [0, 10],
      [10, 20]
    ]
    const scaler = new Normalizer({ norm: 'l1' })
    const expected = scaler.fitTransform(scaler)
    const expectedValueAbove = [
      [-0.5, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [0.33, 0.66]
    ]
 * ```
 */
export class Normalizer extends TransformerMixin {
  norm: string
  /** The number of features seen during fit */
  nFeaturesIn: number

  /** Names of features seen during fit. Only stores feature names if input is a DataFrame */
  featureNamesIn: Array<string>

  constructor({ norm = 'l2' }: NormalizerParams = {}) {
    super()
    this.norm = norm
    this.nFeaturesIn = 0
    this.featureNamesIn = []
  }

  /**
   * Fits a Normalizer to the data
   */
  public fit(X: Scikit2D): Normalizer {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    this.nFeaturesIn = tensorArray.shape[1]
    if (X instanceof dfd.DataFrame) {
      this.featureNamesIn = [...X.columns]
    }
    return this
  }

  /**
   * Transform the data using the Normalizer
   * */
  public transform(X: Scikit2D): tf.Tensor2D {
    assert(isScikit2D(X), 'Data can not be converted to a 2D matrix.')
    const tensorArray = convertToNumericTensor2D(X)
    if (this.norm === 'l1') {
      const means = tf.abs(tensorArray).sum(1).reshape([-1, 1])
      return tensorArray.divNoNan(means)
    }
    if (this.norm === 'l2') {
      const means = tensorArray.square().sum(1).sqrt().reshape([-1, 1])
      return tensorArray.divNoNan(means)
    }
    // max case
    const means = tf.abs(tensorArray).max(1).reshape([-1, 1])
    return tensorArray.divNoNan(means)
  }
}
