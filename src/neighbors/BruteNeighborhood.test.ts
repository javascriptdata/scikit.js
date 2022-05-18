/**
*  @license
* Copyright 2022, JsData. All rights reserved.
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

import { neighborhoodGenericTests } from './neighborhoodGenericTests'
import { BruteNeighborhood } from './BruteNeighborhood'
import { setBackend } from '../tf-singleton'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

neighborhoodGenericTests(
  'BruteNeighborhood',
  async (params) => new BruteNeighborhood(params)
)
