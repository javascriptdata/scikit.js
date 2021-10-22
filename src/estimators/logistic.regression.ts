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

// import '@tensorflow/tfjs-node'
// import { losses, train } from '@tensorflow/tfjs-core'
// import { callbacks } from '@tensorflow/tfjs-layers'
// import { SGD } from './sgd.linear'
// import {
//   initializers,
//   regularizers,
//   Tensor1D,
//   Tensor2D,
// } from '@tensorflow/tfjs-node'
// import { tensor1dConv, tensor2dConv } from 'utils'

// // First pass at a LinearRegression implementation using gradient descent
// // Trying to mimic the API of scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

// export interface LogisticRegressionParams {
//   penalty: 'l1' | 'l2' | 'none'
//   C: number
//   fitIntercept: boolean
// }

// export class LogisticRegression extends SGD {
//   constructor(params: LogisticRegressionParams) {
//     // Assume Binary classification
//     // If we call fit, and it isn't binary then update args
//     super({
//       modelCompileArgs: {
//         optimizer: train.adam(0.1),
//         loss: losses.sigmoidCrossEntropy,
//         metrics: ['accuracy'],
//       },
//       modelFitArgs: {
//         batchSize: 32,
//         epochs: 1000,
//         verbose: 0,
//         callbacks: [
//           callbacks.earlyStopping({ monitor: 'loss', patience: 50 }),
//         ],
//       },
//       denseLayerArgs: {
//         units: 1,
//         useBias: Boolean(params.fitIntercept),
//         kernelInitializer: initializers.zeros(),
//         biasInitializer: initializers.zeros(),
//         kernelRegularizer:
//           params.penalty === 'l2'
//             ? regularizers.l2({ l2: params.C })
//             : params.penalty === 'l1'
//             ? regularizers.l1({ l1: params.C })
//             : undefined,
//       },
//     })
//   }

//   // Placeholder for eventual code that will determine if we need to
//   // do one-hot-encoding
//   async fit(
//     X: Tensor2D | number[][],
//     y: Tensor1D | number[]
//   ): Promise<LogisticRegression> {
//     let XTwoD = tensor2dConv(X)
//     let yOneD = tensor1dConv(y)
//     if (this.model.layers.length === 0) {
//       this.initializeModel(XTwoD.shape[1])
//     }
//     await this.model.fit(XTwoD, yOneD, { ...this.modelFitArgs })
//     return this
//   }
// }
