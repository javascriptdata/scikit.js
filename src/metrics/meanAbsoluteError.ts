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

import { tidy } from '@tensorflow/tfjs-core'
import * as tf from '@tensorflow/tfjs-node'
import {
  assertSameShapeAndType,
  convertToTensor1D_2D,
  ScikitVecOrMatrix,
} from '../utils'

/**
 *
 * @param labels 1D or 2D TensorLike object
 * @param predictions 1D or 2D TensorLike object
 * @returns Tensor that is 0-dimensial (a scalar) or 1-dimensional in the case that
 * we are computing meanAbsoluteError against 2D Tensors.
 */

// computes the mean absolute error between two vectors or two matrics.
// Returns it as a Tensor

export function meanAbsoluteError(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  return tidy(() => {
    let labelsT = convertToTensor1D_2D(labels)
    let predictionsT = convertToTensor1D_2D(predictions)
    assertSameShapeAndType(labelsT, predictionsT)

    return tf.metrics.meanAbsoluteError(labelsT, predictionsT)
  })
}
