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

import { convertToNumericTensor1D } from '../utils'
import { Scikit1D } from '../types'
import { assert, isScikit1D } from '../typesUtils'
import { uniq } from 'lodash'

import { tf } from '../shared/globals'

function assertInputIsWellFormed(labels: Scikit1D, predictions: Scikit1D) {
  assert(isScikit1D(labels), "Labels can't be converted to a 1D Tensor")
  assert(
    isScikit1D(predictions),
    "Predictions can't be converted to a 1D Tensor"
  )

  let labelsT = convertToNumericTensor1D(labels)
  let predictionsT = convertToNumericTensor1D(predictions)
  assert(labelsT.size > 0, 'Must be more than 1 label')
  assert(predictionsT.size > 0, 'Must be more than 1 prediction')
  assert(labelsT.size === predictionsT.size, 'Not the same size arrays')
  return { labelsT, predictionsT }
}

//////////////////////////////////////
// Scoring functions
//////////////////////////////////////

/**
 *
 * ```js
 *const labels = [1, 2, 3, 1]
  const predictions = [1, 2, 4, 4]
  let result = metrics.accuracyScore(labels, predictions)
  console.log(result) // 0.5
 *```
 * @param labels 1D Array-like that are the true values
 * @param predictions 1D Array-like that are your model predictions
 * @returns number
 */
export function accuracyScore(
  labels: Scikit1D,
  predictions: Scikit1D
): number {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = labelsT.equal(predictionsT).sum().div(labelsT.size)
  return result.dataSync()[0]
}

export function precisionScore(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.metrics.precision(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function recallScore(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.metrics.recall(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function r2Score(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )

  const numerator = tf.metrics.meanSquaredError(labelsT, predictionsT)
  const denominator = tf.metrics.meanSquaredError(labelsT, labelsT.mean())

  const result = tf.sub(1, numerator.div(denominator))
  return result.dataSync()[0]
}

//////////////////////////////////////
// Error or Loss functions
//////////////////////////////////////

export function meanAbsoluteError(
  labels: Scikit1D,
  predictions: Scikit1D
): number {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.metrics.meanAbsoluteError(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function meanSquaredError(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.metrics.meanSquaredError(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function meanSquaredLogError(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf
    .square(labelsT.log1p().sub(predictionsT.log1p()))
    .sum()
    .div(labelsT.size)
  return result.dataSync()[0]
}

export function hingeLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.losses.hingeLoss(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function huberLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.losses.huberLoss(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function logLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.losses.logLoss(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function zeroOneLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = tf.sub(1, accuracyScore(labelsT, predictionsT))
  return result.dataSync()[0]
}

//////////////////////////////////////
// Odds and Ends
//////////////////////////////////////

export function confusionMatrix(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )

  const uniqueNumber = uniq(labelsT.dataSync())

  return tf.math
    .confusionMatrix(labelsT, predictionsT, uniqueNumber.length)
    .arraySync()
}

export function rocAucScore(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  // Next steps: This can prob be done faster with tensor magic
  let x = labelsT.arraySync()
  let y = predictionsT.arraySync()
  x.push(1)
  y.push(1)
  let area = 0
  for (let i = 0; i < x.length - 1; i++) {
    area += x[i] * y[i + 1] - x[i + 1] * y[i]
  }
  area -= 1
  return Math.abs(area) / 2
}
