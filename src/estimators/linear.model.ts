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

import '@tensorflow/tfjs-node';
import {
  Tensor,
  Tensor1D,
  Tensor2D,
  losses,
  tensor1d,
  tensor2d,
  train,
} from '@tensorflow/tfjs-core';
import {
  layers,
  sequential,
  Sequential,
  ModelFitArgs,
  ModelCompileArgs,
  callbacks,
} from '@tensorflow/tfjs-layers';
import { tensor2dConv, tensor1dConv } from '../utils';

// First pass at a LinearRegression implementation using gradient descent
// Trying to mimic the API of scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

export interface LinearRegressionParams {
  /**
   * Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
   */
  fitIntercept?: boolean;

  /**
   * The complete list of compile args for the `model.compile` call from tensorflow.js.
   * We aim to provide sensible defaults for LinearRegression.
   */
  modelCompileArgs: ModelCompileArgs;

  /**
   * The complete list of `model.fit` args from Tensorflow.js
   * We aim to provide sensible defaults for LinearRegression.
   */
  modelFitArgs: ModelFitArgs;
}

export default class LinearRegression {
  model: Sequential;
  modelFitArgs: ModelFitArgs;
  modelCompileArgs: ModelCompileArgs;
  fitIntercept: boolean;

  constructor(
    params: LinearRegressionParams = {
      fitIntercept: true,
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
    }
  ) {
    this.model = sequential();
    this.fitIntercept = Boolean(params.fitIntercept);
    this.modelFitArgs = params.modelFitArgs;
    this.modelCompileArgs = params.modelCompileArgs;
  }

  initializeModel(
    inputShape: number,
    useBias = true,
    weightsTensors: Tensor[] = []
  ): void {
    const model = sequential();
    model.add(layers.dense({ inputShape: [inputShape], units: 1, useBias }));
    model.compile(this.modelCompileArgs);
    if (weightsTensors?.length) {
      model.setWeights(weightsTensors);
    }
    this.model = model;
  }

  async fit(X: Tensor2D | number[][], y: Tensor1D | number[]) {
    let XTwoD = tensor2dConv(X);
    let yOneD = tensor1dConv(y);
    if (this.model.layers.length === 0) {
      this.initializeModel(XTwoD.shape[1], this.fitIntercept);
    }
    await this.model.fit(XTwoD, yOneD, { ...this.modelFitArgs });
  }

  importModel(params: {
    coef_: number[];
    intercept_: number;
  }): LinearRegression {
    let myCoef = tensor2d(params.coef_, [params.coef_.length, 1], 'float32');
    let myIntercept = tensor1d([params.intercept_], 'float32');
    this.initializeModel(params.coef_.length, true, [myCoef, myIntercept]);
    return this;
  }

  getParams(): LinearRegressionParams {
    return {
      fitIntercept: this.fitIntercept,
      modelFitArgs: this.modelFitArgs,
      modelCompileArgs: this.modelCompileArgs,
    };
  }

  setParams(params: LinearRegressionParams): LinearRegression {
    this.fitIntercept = Boolean(params.fitIntercept);
    this.modelCompileArgs = params.modelCompileArgs;
    this.modelFitArgs = params.modelFitArgs;
    return this;
  }

  predict(X: Tensor2D | number[][]): Tensor2D {
    let XTwoD = tensor2dConv(X);
    if (this.model.layers.length === 0) {
      throw new RangeError('Need to call "fit" before "predict"');
    }
    return this.model.predict(XTwoD) as Tensor2D;
  }

  get coef_(): Tensor {
    return this.model.getWeights()[0];
  }

  get intercept_(): Tensor {
    return this.model.getWeights()[1];
  }
}
