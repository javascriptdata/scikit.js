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

import { Scikit1D } from '../types'
import { tf } from '../shared/globals'
import { isSeriesInterface } from '../typesUtils'

/*
Next steps:
1. Pass the next 5 tests
*/

/**
 * Encode target labels with value between 0 and n_classes-1.
 * @example
 * ```js
 *  import { LabelEncoder } from 'scikitjs'
 *
 *  const sf = [1, 2, 2, 'boy', 'git', 'git']
    const scaler = new LabelEncoder()
    scaler.fit(sf)
    console.log(scaler.classes) // [1, 2, "boy", "git"]
    scaler.transform([2, 2, "boy"]) // [1, 1, 2]
 * ```
 */
export class LabelEncoder {
  /** Unique classes that we see in this single array of data */
  classes: Array<string | number | boolean>

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'labelencoder'

  constructor() {
    this.classes = []
  }

  convertTo1DArray(X: Scikit1D): Iterable<string | number | boolean> {
    if (isSeriesInterface(X)) {
      return X.values as any[]
    }
    if (X instanceof tf.Tensor) {
      return X.arraySync()
    }
    return X
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
  /**
   * Maps values to unique integer labels between 0 and n_classes-1.
   * @example
   * ```js
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * ```
   */
  public fit(X: Scikit1D): LabelEncoder {
    const arr = this.convertTo1DArray(X)
    const dataSet = Array.from(new Set(arr))
    this.classes = dataSet

    return this
  }

  /**
   * Encode labels with value between 0 and n_classes-1.
   * @example
   * ```js
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * console.log(encoder.transform(["a", "b", "c", "d"]))
   * // [0, 1, 2, 3]
   * ```
   */
  public transform(X: Scikit1D): tf.Tensor1D {
    const arr = this.convertTo1DArray(X)

    const labels = this.classesToMapping(this.classes)
    const encodedData = (arr as any).map((value: any) => {
      let val = labels.get(value)
      return val === undefined ? -1 : val
    })
    return tf.tensor1d(encodedData)
  }

  public fitTransform(X: Scikit1D): tf.Tensor1D {
    return this.fit(X).transform(X)
  }

  /**
   * Inverse transform values back to original values.
   * @example
   * ```js
   * const encoder = new LabelEncoder()
   * encoder.fit(["a", "b", "c", "d"])
   * console.log(encoder.inverseTransform([0, 1, 2, 3]))
   * // ["a", "b", "c", "d"]
   * ```
   */
  public inverseTransform(X: Scikit1D): any[] {
    const arr = this.convertTo1DArray(X)
    const labels = this.classesToMapping(this.classes)
    const invMap = new Map(Array.from(labels, (a) => a.reverse()) as any)

    const tempData = (arr as any).map((value: any) => {
      return invMap.get(value) === undefined ? null : invMap.get(value)
    })
    return tempData
  }
}
