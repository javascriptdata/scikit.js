// /**
// *  @license
// * Copyright 2021, JsData. All rights reserved.
// *
// * This source code is licensed under the MIT license found in the
// * LICENSE file in the root directory of this source tree.

// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// * ==========================================================================
// */

import { SGDClassifier } from '../linear_model/SgdClassifier'
import { getBackend } from '../tf-singleton'

// First pass at a LinearSVC implementation using gradient descent
// Trying to mimic the API of scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearSVC.html

/*
Next steps:
1. Support elasticnet penalty
2. Support tol and maxIter (might need to change sgd.linear)
3. Implement randomState
4. Implement attribute "classes"
5. Pass next 5 scikit-learn tests
*/

export interface LinearSVCParams {
  /** Specify the norm of the penalty. **default = l2** */
  penalty?: 'l1' | 'l2' | 'none'
  /** Inverse of the regularization strength. **default = 1** */
  C?: number
  /** Whether or not the intercept should be estimator not. **default = true** */
  fitIntercept?: boolean
}

/** Builds a linear classification model with associated penalty and regularization
 *
 * @example
 * ```js
 * let X = [
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
    ]
    let y = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    let svc = new LinearSVC()
    await svc.fit(X, y)
 * ```
*/
export class LinearSVC extends SGDClassifier {
  constructor({
    penalty = 'l2',
    C = 1,
    fitIntercept = true
  }: LinearSVCParams = {}) {
    // Assume Binary classification
    // If we call fit, and it isn't binary then update args
    let tf = getBackend()
    super({
      modelCompileArgs: {
        optimizer: tf.train.adam(0.1),
        loss: tf.losses.hingeLoss,
        metrics: ['accuracy']
      },
      modelFitArgs: {
        batchSize: 32,
        epochs: 1000,
        verbose: 0,
        callbacks: [
          tf.callbacks.earlyStopping({ monitor: 'loss', patience: 50 })
        ]
      },
      denseLayerArgs: {
        units: 1,
        useBias: Boolean(fitIntercept),
        activation: 'softmax',
        kernelInitializer: tf.initializers.zeros(),
        biasInitializer: tf.initializers.zeros(),
        kernelRegularizer:
          penalty === 'l2'
            ? tf.regularizers.l2({ l2: C })
            : penalty === 'l1'
            ? tf.regularizers.l1({ l1: C })
            : undefined
      },
      isClassification: true,
      optimizerType: 'adam',
      lossType: 'hingeLoss'
    })
    /** Useful for pipelines and column transformers to have a default name for transforms */
    this.name = 'LinearSVC'
  }
}
