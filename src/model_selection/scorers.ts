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

import { Scikit1D, Scikit2D } from '../types'
import { convertToTensor1D, convertToTensor2D } from '../utils'
import { tf } from '../shared/globals'
type Scalar = tf.Scalar
type Tensor1D = tf.Tensor1D
type Tensor2D = tf.Tensor2D

export function negMeanAbsoluteError(
  this: {
    predict(X: Tensor2D): Tensor1D
  },
  X: Scikit2D,
  y: Scikit1D
) {
  X = convertToTensor2D(X)
  y = convertToTensor1D(y)
  const yPred = this.predict(X)
  return tf.metrics.meanAbsoluteError(y, yPred).neg() as Scalar
}

export function negMeanSquaredError(
  this: {
    predict(X: Tensor2D): Tensor1D
  },
  X: Scikit2D,
  y: Scikit1D
) {
  X = convertToTensor2D(X)
  y = convertToTensor1D(y)
  const yPred = this.predict(X)
  return tf.metrics.meanSquaredError(y, yPred).neg() as Scalar
}

export function accuracy(
  this: {
    predict(X: Tensor2D): Tensor1D
  },
  X: Scikit2D,
  y: Scikit1D
) {
  X = convertToTensor2D(X)
  y = convertToTensor1D(y)
  const yPred = this.predict(X)
  return tf.equal(y, yPred).sum().div(y.shape[0]) as Scalar
}
