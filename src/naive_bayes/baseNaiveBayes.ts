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
import { polyfillUnique } from '../tfUtils'
import { tf } from '../shared/globals'
import { Scikit1D, Scikit2D } from '../types'
import { convertToNumericTensor2D, convertToTensor1D } from '../utils'
import { Tensor1D } from '@tensorflow/tfjs-core'
import Serialize from '../serialize'

export interface NaiveBayesParams {
  /**
   * Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
   */
  priors?: Scikit1D

  /**
   * Portion of the largest variance of all features that is added to variances for calculation stability.
   * Default value is 1e-9
   */
  varSmoothing?: number
}

export abstract class BaseNaiveBayes extends Serialize {
  priors?: tf.Tensor1D
  varSmoothing: number

  public classes: tf.Tensor1D
  public means: tf.Tensor1D[]
  public variances: tf.Tensor1D[]

  constructor(params: NaiveBayesParams = {}) {
    super()
    this.classes = tf.tensor1d([])
    this.means = []
    this.variances = []
    if (params.priors) {
      this.priors = convertToTensor1D(params.priors)
    }
    this.varSmoothing = params.varSmoothing ? params.varSmoothing : 1e-9
  }

  /**
   * Train the model by calculating the mean and variance of sample distribution.
   * @param X
   * @param y
   * @returns
   */
  public async fit(X: Scikit2D, y: Scikit1D) {
    const features = convertToNumericTensor2D(X)
    const labels = convertToTensor1D(y)

    const { values, meansByLabel, variancesByLabel } = tf.tidy(() => {
      polyfillUnique(tf)
      const meansByLabel: Tensor1D[] = []
      const variancesByLabel: Tensor1D[] = []

      // Get the list of unique labels
      const { values } = tf.unique(labels)

      const { variance } = tf.moments(features, 0)
      const epsilon = variance.max().mul(this.varSmoothing)

      tf.unstack(values).forEach((c: tf.Tensor) => {
        const mask = tf.equal(labels, c).toFloat()
        const numInstances = tf.sum(mask)
        const mean = tf
          .mul(features, mask.expandDims(1))
          .sum(0)
          .div(numInstances)
        const variance = tf
          .sub(features, mean)
          .mul(mask.expandDims(1))
          .pow(2)
          .sum(0)
          .div(numInstances)
          .add(epsilon)

        meansByLabel.push(mean as Tensor1D)
        variancesByLabel.push(variance as Tensor1D)
      })

      return { values, meansByLabel, variancesByLabel }
    })

    // Unique labels this model have learned
    this.classes = values
    this.means = meansByLabel
    this.variances = variancesByLabel

    return this
  }

  /**
   * Predict the probability of samples assigned to each observed label.
   * @param X
   * @returns {tf.Tensor} Probabilities
   */
  public predictProba(X: Scikit2D) {
    const features = convertToNumericTensor2D(X)

    const probabilities = tf.tidy(() => {
      let probs: tf.Tensor1D[] = []
      this.classes.unstack().forEach((_, idx) => {
        // Get the mean for this label
        const mean = this.means[idx]
        const variance = this.variances[idx]

        const prob = this.kernel(features, mean, variance)
        probs.push(prob as tf.Tensor1D)
      })

      const withoutPriors = tf.stack(probs, 1) as tf.Tensor2D
      if (this.priors) {
        return withoutPriors.mul(this.priors)
      } else {
        return withoutPriors
      }
    })

    return probabilities
  }

  /**
   * Predict the labels assigned to each sample
   * @param X
   * @returns {tf.Tensor} Labels
   */
  public predict(X: Scikit2D) {
    const probs = this.predictProba(X)
    return probs.argMax(1)
  }

  /**
   * Kernel function to calculate posterior probability, which should be implemented in the subclass.
   * @param features
   * @param mean
   * @param variance
   */
  protected abstract kernel(
    features: tf.Tensor2D,
    mean: tf.Tensor1D,
    variance: tf.Tensor1D
  ): tf.Tensor1D

  public toJson(): string {
    const jsonClass = JSON.parse(super.toJson() as string)

    if (this.priors) {
      jsonClass.priors = this.priors.arraySync()
    }
    jsonClass.classes = this.classes.arraySync()
    jsonClass.means = this.means.map((t: Tensor1D) => t.arraySync())
    jsonClass.variances = this.variances.map((v: Tensor1D) => v.arraySync())
    return JSON.stringify(jsonClass)
  }

  public fromJson(model: string) {
    const jsonModel = JSON.parse(model)

    if (jsonModel.priors) {
      jsonModel.priors = tf.tensor(jsonModel.priors)
    }
    jsonModel.classes = tf.tensor(jsonModel.classes)

    const means = []
    for (const wMeans of jsonModel.means) {
      means.push(tf.tensor(wMeans))
    }
    const variances = []
    for (const variance of jsonModel.variances) {
      variances.push(tf.tensor(variance))
    }
    jsonModel.means = means
    jsonModel.variances = variances
    return Object.assign(this, jsonModel) as this
  }
}
