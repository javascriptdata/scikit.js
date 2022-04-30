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

import { SGDRegressor } from './SgdRegressor'
import { getBackend } from '../tf-singleton'

// RidgeRegression implementation using gradient descent
// This is a placeholder until we can do an analytic solution instead
// https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

export interface RidgeRegressionParams {
  /** Whether or not the intercept should be estimator not. **default = true** */
  fitIntercept?: boolean

  /**Constant that multiplies the penalty terms. **default = .01**  */
  alpha?: number
}

/** Linear least squares with l2 regularization. */
export class RidgeRegression extends SGDRegressor {
  constructor({
    fitIntercept = true,
    alpha = 0.01
  }: RidgeRegressionParams = {}) {
    let tf = getBackend()
    super({
      modelCompileArgs: {
        optimizer: tf.train.adam(0.1),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
      },
      modelFitArgs: {
        batchSize: 32,
        epochs: 1000,
        verbose: 0,
        callbacks: [
          tf.callbacks.earlyStopping({ monitor: 'mse', patience: 50 })
        ]
      },
      denseLayerArgs: {
        units: 1,
        kernelRegularizer: tf.regularizers.l2({ l2: alpha }),
        useBias: Boolean(fitIntercept)
      },
      optimizerType: 'adam',
      lossType: 'meanSquaredError'
    })
    /** Useful for pipelines and column transformers to have a default name for transforms */
    this.name = 'RidgeRegression'
  }
}
