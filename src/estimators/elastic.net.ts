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
import { regularizers } from '@tensorflow/tfjs-node'

// First pass at a ElasticNet implementation using gradient descent
// Trying to mimic the API of https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet

export interface ElasticNetParams {
  alpha: number
  l1Ratio: number
  fitIntercept: number
}

export class ElasticNet extends SGD {
  constructor(params: ElasticNetParams) {
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
        callbacks: [callbacks.earlyStopping({ monitor: 'mse', patience: 50 })],
      },
      denseLayerArgs: {
        units: 1,
        kernelRegularizer: regularizers.l1l2({
          l1: params.alpha * params.l1Ratio,
          l2: 0.5 * params.alpha * (1 - params.l1Ratio),
        }),
        useBias: Boolean(params.fitIntercept),
      },
    })
  }
}
