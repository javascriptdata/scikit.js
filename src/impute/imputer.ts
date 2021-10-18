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

import { convertToNumericTensor1D_2D, meanIgnoreNaNSafe } from 'utils'
import { ScikitVecOrMatrix } from 'types'
import { quantile } from 'simple-statistics'

type Strategy = 'mean' | 'median' | 'mostFrequent' | 'constant'

export class SimpleImputer {
  missingValues: number | string | null | undefined
  strategy: Strategy
  fillValue: any = undefined
  constructor(
    missingValues: number | string | null | undefined = NaN,
    strategy: Strategy = 'mean',
    fillValue: any = null
  ) {
    this.missingValues = missingValues
    this.strategy = strategy
    this.fillValue = fillValue
  }

  fit(X: ScikitVecOrMatrix) {
    // Fill with value passed into fillValue argument
    if (this.strategy === 'constant') {
      return this
    }
    if (this.strategy === 'mean') {
      const newTensor = convertToNumericTensor1D_2D(X)
      const mean = meanIgnoreNaNSafe(newTensor, 0)
      this.fillValue = mean
    }
    if (this.strategy === 'mostFrequent') {
      console.log('here')
    }
    if (this.strategy === 'median') {
      console.log('Support median')
    }
    throw new Error(
      `Strategy ${this.strategy} is unsupported. Supported strategies are 'mean', 'median', 'mostFrequent', and 'constant'`
    )
  }

  transform(X: ScikitVecOrMatrix) {}
}
