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

import '@tensorflow/tfjs-node'
import { losses, train } from '@tensorflow/tfjs-core'
import { callbacks } from '@tensorflow/tfjs-layers'
import { SGD } from './sgd.linear'

/**
 * LinearRegression implementation using gradient descent
 * We aim to mimic the API of scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
 *
 * The heavy lifting is done in the SGD class.
 * This simply provides sane defaults for a Linear Regression.
 *
 * Potentially we eventually make this class do the "exact" solution using the normal equations
 * https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/
 *
 * In order to do that though, we'd need to do a matrix inversion and tensorflow.js doesn't currently support it.
 * Moreover, for big input / output combinations SGD is faster than doing the matrix inversion anyway.
 * So even if we do eventually do the exact solution, we should then call whichever version is faster (SGD vs Exact)
 * depending on the size of the input.
 */

export interface LinearRegressionParams {
  /**
   * Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
   */
  fitIntercept?: boolean
}

export class LinearRegression extends SGD {
  constructor(params: LinearRegressionParams = { fitIntercept: true }) {
    super({
      modelCompileArgs: {
        optimizer: train.adam(0.1),
        loss: losses.meanSquaredError,
        metrics: ['mse'],
      },
      modelFitArgs: {
        batchSize: 32,
        epochs: 1000,
        verbose: 0,
        callbacks: [callbacks.earlyStopping({ monitor: 'mse', patience: 30 })],
      },
      denseLayerArgs: {
        units: 1,
        useBias: Boolean(params.fitIntercept),
      },
    })
  }
}