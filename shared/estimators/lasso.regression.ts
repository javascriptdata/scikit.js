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
import { SGD } from './sgd.linear'
import { tf } from '../../globals'

// First pass at a LassoRegression implementation using gradient descent
// Trying to mimic the API of https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso

export interface LassoParams {
  fitIntercept: boolean
  alpha: number
}

export class LassoRegression extends SGD {
  constructor({ fitIntercept, alpha }: LassoParams) {
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
        kernelRegularizer: tf.regularizers.l1({ l1: alpha }),
        useBias: Boolean(fitIntercept)
      }
    })
  }
}
