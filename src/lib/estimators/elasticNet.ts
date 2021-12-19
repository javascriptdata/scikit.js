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

import { losses, train } from '@tensorflow/tfjs-core'
import { callbacks } from '@tensorflow/tfjs-layers'
import { SGD } from './sgdLinear'
import { tf } from '../../globals'

// First pass at a ElasticNet implementation using gradient descent
// Trying to mimic the API of https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

export interface ElasticNetParams {
  /**Constant that multiplies the penalty terms. **default = .01**  */
  alpha?: number

  /**The ElasticNet mixing parameter. **default = .5** */
  l1Ratio?: number
  /** Whether or not the intercept should be estimator not. **default = true** */
  fitIntercept?: boolean
}

/**
 * Linear regression with combined L1 and L2 priors as regularizer.
 */
export class ElasticNet extends SGD {
  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'elasticnet'

  constructor({
    alpha = 1,
    l1Ratio = 0.5,
    fitIntercept = true
  }: ElasticNetParams = {}) {
    super({
      modelCompileArgs: {
        optimizer: train.adam(0.1),
        loss: losses.meanSquaredError,
        metrics: ['mse']
      },
      modelFitArgs: {
        batchSize: 32,
        epochs: 1000,
        verbose: 0,
        callbacks: [callbacks.earlyStopping({ monitor: 'mse', patience: 50 })]
      },
      denseLayerArgs: {
        units: 1,
        kernelRegularizer: tf.regularizers.l1l2({
          l1: alpha * l1Ratio,
          l2: 0.5 * alpha * (1 - l1Ratio)
        }),
        useBias: Boolean(fitIntercept)
      }
    })
  }
}
