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

import { convertToNumericTensor1D, convertTensorToInputType } from '../utils'
import { Scikit1D, Strategy } from '../types'
import { tensorMean } from '../math'
import { median, modeFast } from 'simple-statistics'
import { where } from '@tensorflow/tfjs-core'
import { TransformerMixin } from '../mixins'

export default class SimpleImputer extends TransformerMixin {
  missingValues: number | string | null | undefined
  fillValue: number | string = 0
  strategy: Strategy

  constructor(
    strategy: Strategy = 'mean',
    fillValue: number | string = 0,
    missingValues: number | string | null | undefined = NaN
  ) {
    super()
    this.missingValues = missingValues
    this.strategy = strategy
    this.fillValue = fillValue
  }

  fit(data: Scikit1D): SimpleImputer {
    const newTensor = convertToNumericTensor1D(data)
    // Fill with value passed into fillValue argument
    if (this.strategy === 'constant') {
      return this
    }
    if (this.strategy === 'mean') {
      const mean = tensorMean(newTensor, 0, true)
      this.fillValue = mean.dataSync()[0]
      return this
    }
    if (this.strategy === 'mostFrequent') {
      const array = newTensor.arraySync()
      this.fillValue = modeFast(array as number[])
      return this
    }
    if (this.strategy === 'median') {
      const array = newTensor.arraySync()
      this.fillValue = median(array as number[])
      return this
    }
    throw new Error(
      `Strategy ${this.strategy} is unsupported. Supported strategies are 'mean', 'median', 'mostFrequent', and 'constant'`
    )
  }

  transform(data: Scikit1D): Scikit1D {
    const newTensor = convertToNumericTensor1D(data)
    const filledTensor = where(newTensor.isNaN(), this.fillValue, newTensor)
    return convertTensorToInputType(filledTensor, data) as Scikit1D
  }
}
