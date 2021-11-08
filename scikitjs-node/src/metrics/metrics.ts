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

import { metrics, losses, math, sub, square } from '@tensorflow/tfjs-node'
import { convertToNumericTensor1D } from '../utils'
import { Scikit1D } from '../types'
import { assert, isScikit1D } from '../types.utils'
import { uniq } from 'lodash'

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
  const result = metrics.precision(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function recallScore(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = metrics.recall(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function r2Score(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )

  const numerator = metrics.meanSquaredError(labelsT, predictionsT)
  const denominator = metrics.meanSquaredError(labelsT, labelsT.mean())

  const result = sub(1, numerator.div(denominator))
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
  const result = metrics.meanAbsoluteError(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function meanSquaredError(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = metrics.meanSquaredError(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function meanSquaredLogError(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = square(labelsT.log1p().sub(predictionsT.log1p()))
    .sum()
    .div(labelsT.size)
  return result.dataSync()[0]
}

export function hingeLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = losses.hingeLoss(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function huberLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = losses.huberLoss(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function logLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = losses.logLoss(labelsT, predictionsT)
  return result.dataSync()[0]
}

export function zeroOneLoss(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  const result = sub(1, accuracyScore(labelsT, predictionsT))
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

  return math
    .confusionMatrix(labelsT, predictionsT, uniqueNumber.length)
    .arraySync()
}

export function rocAucScore(labels: Scikit1D, predictions: Scikit1D) {
  const { labelsT, predictionsT } = assertInputIsWellFormed(
    labels,
    predictions
  )
  // Todo: This can prob be done faster with tensor magic
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
