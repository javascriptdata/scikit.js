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
import {
  Tensor,
  Tensor1D,
  Tensor2D,
  tensor1d,
  tensor2d,
} from '@tensorflow/tfjs-core'
import {
  layers,
  sequential,
  Sequential,
  ModelFitArgs,
  ModelCompileArgs,
} from '@tensorflow/tfjs-layers'
import { tensor2dConv, tensor1dConv } from '../utils'
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core'

/**
 * SGD is a thin Wrapper around Tensorflow's model api with a single dense layer.
 * With this base class and different error functions / regularizers we can
 * create SGD solvers for LinearRegression, RidgeRegression, LassoRegression,
 * ElasticNet, LogisticRegression and many more.
 */

/**
 * Parameters for SGD
 */
export interface SGDParams {
  /**
   * The complete list of compile args for the `model.compile` call from tensorflow.js.
   * We aim to provide sensible defaults depending on the regressor / classifier.
   * An example call might look like
   *  model.compile({
        optimizer: train.adam(0.1),
        loss: losses.meanSquaredError,
        metrics: ['mse'],
      })
   */
  modelCompileArgs: ModelCompileArgs

  /**
   * The complete list of `model.fit` args from Tensorflow.js
   * We aim to provide sensible defaults depending on the regressor / classifier.
   * An example call might look like
   *  model.fit(
        batchSize: 32,
        epochs: 1000,
        verbose: 0,
        callbacks: [callbacks.earlyStopping({ monitor: 'mse', patience: 50 })],
      })
   */
  modelFitArgs: ModelFitArgs

  /**
   * The arguments for a single dense layer in tensorflow. This also defaults to 
   * different settings based on the regressor / classifier. An example dense layer
   * might look like.
   *  const model = sequential()
      model.add(
        layers.dense({ inputShape: [100], 
        units: 1,
        useBias: true,
      })
      )
   */
  denseLayerArgs: DenseLayerArgs
}

export class SGD {
  model: Sequential
  modelFitArgs: ModelFitArgs
  modelCompileArgs: ModelCompileArgs
  denseLayerArgs: DenseLayerArgs

  constructor(params: SGDParams) {
    this.model = sequential()
    this.modelFitArgs = params.modelFitArgs
    this.modelCompileArgs = params.modelCompileArgs
    this.denseLayerArgs = params.denseLayerArgs
  }

  /**
   * Creates the tensorflow model. Because the model contains only
   * one dense layer, we must pass the inputShape to that layer.
   * That inputShape is only known at "runtime" ie... when we call `fit(X, y)`
   * that first time. The inputShape is effectively `X.shape[1]`
   *
   * This function runs after that first call to fit or when pass in modelWeights.
   * That can come up if we train a model in python, and simply want to copy over the
   * weights to this JS version so we can deploy on browsers / phones.
   * @returns {void}
   */

  initializeModel(inputShape: number, weightsTensors: Tensor[] = []): void {
    const model = sequential()
    model.add(
      layers.dense({ inputShape: [inputShape], ...this.denseLayerArgs })
    )
    model.compile(this.modelCompileArgs)
    if (weightsTensors?.length) {
      model.setWeights(weightsTensors)
    }
    this.model = model
  }

  /**
   * Similar to scikit-learn, this trains a model to predict y, from X.
   * Even in the case where we predict a single output vector,
   * the predictions are a 2D matrix (albeit a single column in a 2D Matrix).
   *
   * This is to facilitate the case where we predict multiple targets, or in the case
   * of classification where we are predicting a 2D Matrix of probability class labels.
   * @param {Tensor2D | number[][]} X The 2DTensor / 2D Array that you wish to use as a training matrix
   * @param {Tensor1D | number[]} y The output vector that you wish to predict
   *
   * @returns {Promise<SGD>} Returns the predictions.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * await lr.fit(X, y);
   * // lr model weights have been updated
   */

  async fit(X: Tensor2D | number[][], y: Tensor1D | number[]): Promise<SGD> {
    let XTwoD = tensor2dConv(X)
    let yOneD = tensor1dConv(y)
    if (this.model.layers.length === 0) {
      this.initializeModel(XTwoD.shape[1])
    }
    await this.model.fit(XTwoD, yOneD, { ...this.modelFitArgs })
    return this
  }

  /**
   * This aims to be a bridge to scikit-learn Estimators, where users can train
   * models over in scikit-learn and then ship the coefficients into the proper
   * Estimator on the Scikit.js side. This can be useful if the python version is faster
   * to train, but we still need a JS version because we wish to ship to mobile or browsers.
   *
   * @param {{ coef_: number[]; intercept_: number }} params The object that contains the model parameters,
   * coef_, and intercept_ that we need for our model.
   *
   * @returns {SGD} Returns the predictions.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * lr.importModel({coef_ : [1.2, 2.3], intercept_: 10.0});
   * // lr model weights have been updated
   */

  importModel(params: { coef_: number[]; intercept_: number }): SGD {
    let myCoef = tensor2d(params.coef_, [params.coef_.length, 1], 'float32')
    let myIntercept = tensor1d([params.intercept_], 'float32')
    this.initializeModel(params.coef_.length, [myCoef, myIntercept])
    return this
  }

  /**
   * Similar to scikit-learn, this returns the object of configuration params for SGD
   * @returns {SGDParams} Returns an object of configuration params.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * lr.getParams()
   * // => 
    {
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
        useBias: true,
      }
    }
   */

  getParams(): SGDParams {
    return {
      modelFitArgs: this.modelFitArgs,
      modelCompileArgs: this.modelCompileArgs,
      denseLayerArgs: this.denseLayerArgs,
    }
  }

  /**
   * Similar to scikit-learn, this returns the object of configuration params for SGD
   * @returns {SGDParams} Returns an object of configuration params.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * lr.setParams({
      modelFitArgs: {
        batchSize: 100,
        epochs: -1,
        verbose: 1,
      })
   */

  setParams(params: SGDParams): SGD {
    this.modelCompileArgs = params.modelCompileArgs
    this.modelFitArgs = params.modelFitArgs
    this.denseLayerArgs = params.denseLayerArgs
    return this
  }

  /**
   * Similar to scikit-learn, this returns a Tensor2D (2D Matrix) of predictions.
   * Even in the case where we predict a single output vector,
   * the predictions are a 2D matrix (albeit a single column in a 2D Matrix).
   *
   * This is to facilitate the case where we predict multiple targets, or in the case
   * of classification where we are predicting a 2D Matrix of probability class labels.
   * @param {Tensor2D | number[][]} X The 2DTensor / 2D Array that you wish to run through
   * your model and make predictions.
   *
   * @returns {Tensor2D} Returns the predictions.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * await lr.fit(X, y);
   * lr.predict(X)
   * // => tensor2d([[ 4.5, 10.3, 19.1, 0.22 ]])
   */

  predict(X: Tensor2D | number[][]): Tensor2D {
    let XTwoD = tensor2dConv(X)
    if (this.model.layers.length === 0) {
      throw new RangeError('Need to call "fit" before "predict"')
    }
    return this.model.predict(XTwoD) as Tensor2D
  }

  /**
   * Similar to scikit-learn, this returns the coefficients of our linear model.
   * The return type is a 1D matrix (technically a Tensor1D) if we predict a single output.
   * It's a 2D matrix (Tensor2D) if we predict a regression task with multiple outputs or
   * a classification task with multiple class labels.
   * @returns {Tensor1D | Tensor2D} Returns the coefficients.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * await lr.fit(X, [1,2,3]);
   * lr.coef_
   * // => tensor1d([[ 1.2, 3.3, 1.1, 0.2 ]])
   * 
   * await lr.fit(X, [ [1,2], [3,4], [5,6] ]);
   * lr.coef_
   * // => tensor2d([ [1.2, 3.3], [3.4, 5.6], [4.5, 6.7] ])

   */

  get coef_(): Tensor1D | Tensor2D {
    const modelWeights = this.model.getWeights()
    if (modelWeights.length === 0) {
      return tensor2d([])
    }
    let coefficients = modelWeights[0]
    if (coefficients.shape[1] === 1) {
      return coefficients.reshape([coefficients.shape[0]]) as Tensor1D
    }
    return coefficients as Tensor2D
  }

  /**
   * Similar to scikit-learn, this returns the intercept of our linear model.
   * The return type is always a Tensor1D (a vector).
   * Normally we'd just return a single number but in the case
   * of multiple regression (multiple output targets) we'd need
   * a vector to store all the intercepts,
   * @returns {number | Tensor1D} Returns the intercept.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * await lr.fit(X, [1,2,3]);
   * lr.intercept_
   * // => 4.5
   *
   *
   * lr = new LinearRegression()
   * await lr.fit(X, [ [1,2,3], [4,5,6] ]);
   * lr.intercept_
   * // => tensor1d([1.2, 2.3])
   */
  get intercept_(): number | Tensor1D {
    const modelWeights = this.model.getWeights()
    if (modelWeights.length < 2) {
      return 0.0
    }
    let intercept = modelWeights[1] as Tensor1D
    if (intercept.size === 1) {
      return intercept.arraySync()[0]
    }

    return intercept
  }
}
