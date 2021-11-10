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

import { losses, train } from '@tensorflow/tfjs-core'
import { callbacks } from '@tensorflow/tfjs-layers'
import { SGD } from './sgd.linear'
import { tf } from '../globals'

// First pass at a LogisticRegression implementation using gradient descent
// Trying to mimic the API of scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

export interface LogisticRegressionParams {
  penalty?: 'l1' | 'l2' | 'none'
  C?: number
  fitIntercept?: boolean
}

export class LogisticRegression extends SGD {
  constructor({
    penalty = 'l2',
    C = 1,
    fitIntercept = true
  }: LogisticRegressionParams = {}) {
    if (C === undefined) {
      C = 1
    }
    // Assume Binary classification
    // If we call fit, and it isn't binary then update args
    super({
      modelCompileArgs: {
        optimizer: train.adam(0.1),
        loss: losses.softmaxCrossEntropy,
        metrics: ['accuracy']
      },
      modelFitArgs: {
        batchSize: 32,
        epochs: 1000,
        verbose: 0,
        callbacks: [callbacks.earlyStopping({ monitor: 'loss', patience: 50 })]
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
      isClassification: true
    })
  }
}
