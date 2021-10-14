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

import { Tensor1D, tidy } from '@tensorflow/tfjs-core'
import * as tf from '@tensorflow/tfjs-node'
import {
  assertSameShape,
  convertToTensor1D_2D,
  Scikit1D,
  ScikitVecOrMatrix,
} from '../utils'

/**
 *
 * @param labels 1D or 2D TensorLike object
 * @param predictions 1D or 2D TensorLike object
 * @returns Tensor that is 0-dimensial (a scalar) or 1-dimensional in the case that
 * we are computing meanAbsoluteError against 2D Tensors.
 */

export function meanAbsoluteError(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  return tidy(() => {
    let labelsT = convertToTensor1D_2D(labels)
    let predictionsT = convertToTensor1D_2D(predictions)
    assertSameShape(labelsT, predictionsT)

    return tf.metrics.meanAbsoluteError(labelsT, predictionsT)
  })
}

export function meanSquaredError(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  return tidy(() => {
    let labelsT = convertToTensor1D_2D(labels)
    let predictionsT = convertToTensor1D_2D(predictions)
    assertSameShape(labelsT, predictionsT)

    return tf.metrics.meanSquaredError(labelsT, predictionsT)
  })
}

/*
  The total number of complete matches

  a.equal(b).sum()
*/
export function accuracyScore(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)

  return labelsT.equal(predictionsT).sum()
}

/*

  Calculate AUC
  x -> 1d shape of x coordinates
  y -> 1d shape of y coordinates

  monotically increasing for both (assert on that)
*/
export function auc(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  // Do slice magic
}

/*
  yTrue.add(1).log().sub(yPred.add(1).log()).square().sum()
*/
export function meanSquaredLogError(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)

  tf.square(labelsT.log1p().sub(predictionsT.log1p())).sum()
}

export function confusionMatrix(labels: Scikit1D, predictions: Scikit1D) {
  let labelsT = convertToTensor1D_2D(labels) as Tensor1D
  let predictionsT = convertToTensor1D_2D(predictions) as Tensor1D
  assertSameShape(labelsT, predictionsT)
  return tf.math.confusionMatrix(
    labelsT,
    predictionsT,
    tf.unique(labelsT).values.size
  )
}

export function hingeLoss(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)
  return tf.losses.hingeLoss(labelsT, predictionsT)
}

export function huberLoss(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)
  return tf.losses.huberLoss(labelsT, predictionsT)
}

export function logLoss(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)
  return tf.losses.logLoss(labelsT, predictionsT)
}

export function precisionScore(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)
  return tf.metrics.precision(labelsT, predictionsT)
}

export function recallScore(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)
  return tf.metrics.recall(labelsT, predictionsT)
}

export function zeroOneLoss(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  return tf.sub(1, accuracyScore(labels, predictions))
}

export function r2Score(
  labels: ScikitVecOrMatrix,
  predictions: ScikitVecOrMatrix
) {
  let labelsT = convertToTensor1D_2D(labels)
  let predictionsT = convertToTensor1D_2D(predictions)
  assertSameShape(labelsT, predictionsT)

  const numerator = tf.metrics.meanSquaredError(labelsT, predictionsT)
  const denominator = tf.metrics.meanSquaredError(labelsT, labelsT.mean())

  return tf.sub(1, numerator.div(denominator))
}
