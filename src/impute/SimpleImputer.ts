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

import { convertToNumericTensor2D, convertToTensor2D } from '../utils'
import { Scikit2D, Tensor2D, Tensor1D, TensorLike } from '../types'
import { tensorMean } from '../math'
import { median } from 'mathjs'
import { modeFast } from 'simple-statistics'
import { TransformerMixin } from '../mixins'
import { getBackend } from '../tf-singleton'

/*
Next steps:
1. Make SimpleImputer work with strings
*/

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isEmpty(value: any) {
  return (
    value === undefined ||
    value === null ||
    (isNaN(value) && typeof value !== 'string')
  )
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function removeMissingValuesFromArray(arr: any[]) {
  const values = arr.filter((val) => {
    return !isEmpty(val)
  })
  return values
}

export interface SimpleImputerParams {
  /** The strategy you'd use to impute missing values. "mean" means
   * fill missing values with the mean. Likewise for "median" and "mostFrequent".
   * Use "constant" if you'd like to pass in a "fillValue" and use that to fill
   * missing values. **default = "mean"**
   */
  strategy?: 'mean' | 'median' | 'mostFrequent' | 'constant'

  /** If you choose "constant" pick a value that you'd
   * like to use to fill the missing values. **default = undefined**
   */
  fillValue?: string | number | undefined
  /** This value is the actual missing value. **default = NaN** */
  missingValues?: number | string | null | undefined
}

export class SimpleImputer extends TransformerMixin {
  missingValues: number | string | null | undefined
  fillValue: string | number | undefined
  strategy: 'mean' | 'median' | 'mostFrequent' | 'constant'

  statistics: Tensor1D

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'SimpleImputer'

  tf: any
  constructor({
    strategy = 'mean',
    fillValue = undefined,
    missingValues = NaN
  }: SimpleImputerParams = {}) {
    super()
    this.tf = getBackend()
    this.missingValues = missingValues
    this.strategy = strategy
    this.fillValue = fillValue
    this.statistics = this.tf.tensor1d([])
  }

  public fit(X: Scikit2D): SimpleImputer {
    // Fill with value passed into fillValue argument
    if (this.strategy === 'constant') {
      return this
    }
    if (this.strategy === 'mean') {
      const newTensor = convertToNumericTensor2D(X)
      const mean = tensorMean(newTensor, 0, true)
      this.statistics = mean as Tensor1D
      return this
    }
    if (this.strategy === 'mostFrequent') {
      const newTensor = convertToNumericTensor2D(X)
      const mostFrequents = newTensor
        .transpose<Tensor2D>()
        .arraySync()
        .map((arr: number[] | string[]) =>
          modeFast(removeMissingValuesFromArray(arr))
        )
      this.statistics = this.tf.tensor1d(mostFrequents)
      return this
    }
    if (this.strategy === 'median') {
      const newTensor = convertToNumericTensor2D(X)
      const medians = newTensor
        .transpose<Tensor2D>()
        .arraySync()
        .map((arr: number[] | string[]) =>
          median(removeMissingValuesFromArray(arr))
        )
      this.statistics = this.tf.tensor1d(medians)
      return this
    }
    throw new Error(
      `Strategy ${this.strategy} is unsupported. Supported strategies are 'mean', 'median', 'mostFrequent', and 'constant'`
    )
  }

  public transform(X: Scikit2D): Tensor2D {
    if (this.strategy === 'constant') {
      const newTensor = convertToTensor2D(X)
      if (this.fillValue === undefined) {
        if (newTensor.dtype !== 'string') {
          return this.tf.where(
            newTensor.isNaN(),
            0,
            newTensor as unknown as TensorLike
          ) as Tensor2D
        } else {
          return this.tf.where(
            newTensor.isNaN(),
            'missing_value',
            newTensor as unknown as TensorLike
          ) as Tensor2D
        }
      }
      return this.tf.where(
        newTensor.isNaN(),
        this.fillValue,
        newTensor as unknown as TensorLike
      ) as Tensor2D
    }

    // Not strategy constant
    const newTensor = convertToNumericTensor2D(X)
    return this.tf.where(
      newTensor.isNaN(),
      this.statistics.reshape([1, -1]) as Tensor2D,
      newTensor
    ) as Tensor2D
  }
}
