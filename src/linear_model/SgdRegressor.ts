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
import {
  convertToNumericTensor1D_2D,
  convertToNumericTensor2D
} from '../utils'
import {
  Scikit2D,
  Scikit1D,
  OptimizerTypes,
  LossTypes,
  Tensor1D,
  Tensor2D,
  Tensor,
  ModelCompileArgs,
  ModelFitArgs
} from '../types'
import { RegressorMixin } from '../mixins'
import { getBackend } from '../tf-singleton'

/**
 * SGD is a thin Wrapper around Tensorflow's model api with a single dense layer.
 * With this base class and different error functions / regularizers we can
 * create SGD solvers for LinearRegression, RidgeRegression, LassoRegression,
 * ElasticNet, LogisticRegression and many more.
 */

/**
 * Parameters for SGD
 */
export interface SGDRegressorParams {
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
  denseLayerArgs: any //DenseLayerArgs

  /**
   * This class specifies that we are building a linear model that uses SGD. But there still is the
   * question, "Is this model performing classification or regression"? This argument answers that
   * definitely. It's a boolean that is true when the model aims to perform classification
   */

  isClassification?: boolean

  optimizerType: OptimizerTypes

  lossType: LossTypes

  randomState?: number
}

export class SGDRegressor extends RegressorMixin {
  model: any //this.tf.Sequential
  modelFitArgs: ModelFitArgs
  modelCompileArgs: ModelCompileArgs
  denseLayerArgs: any //DenseLayerArgs
  isMultiOutput: boolean
  optimizerType: OptimizerTypes
  lossType: LossTypes
  randomState?: number

  constructor({
    modelFitArgs,
    modelCompileArgs,
    denseLayerArgs,
    optimizerType,
    lossType,
    randomState
  }: SGDRegressorParams) {
    super()
    this.tf = getBackend()
    this.model = this.tf.sequential()
    this.modelFitArgs = modelFitArgs
    this.modelCompileArgs = modelCompileArgs
    this.denseLayerArgs = denseLayerArgs
    this.isMultiOutput = false
    this.optimizerType = optimizerType
    this.lossType = lossType
    this.randomState = randomState
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

  initializeModel(
    X: Tensor2D,
    y: Tensor1D | Tensor2D,
    weightsTensors: Tensor[] = []
  ): void {
    this.denseLayerArgs.units = y.shape.length === 1 ? 1 : y.shape[1]
    const model = this.tf.sequential()
    let denseLayerArgs = {
      inputShape: [X.shape[1]],
      ...this.denseLayerArgs
    }
    // If randomState is set, then use it to set the args in this layer
    if (this.randomState) {
      denseLayerArgs.kernelInitializer = this.tf.initializers.glorotUniform({
        seed: this.randomState
      })
    }
    model.add(this.tf.layers.dense(denseLayerArgs))
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
   * @param {Scikit2D} X The 2DTensor / 2D Array that you wish to use as a training matrix
   * @param {ScikitVecOrMatrix} y Either 1D or 2D array / Tensor that you wish to predict
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

  public async fit(
    X: Scikit2D,
    y: Scikit1D | Scikit2D
  ): Promise<SGDRegressor> {
    let XTwoD = convertToNumericTensor2D(X)
    let yOneD = convertToNumericTensor1D_2D(y)
    if (yOneD.shape.length > 1) {
      this.isMultiOutput = true
    }

    if (this.model.layers.length === 0) {
      this.initializeModel(XTwoD, yOneD)
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
   * @param {{ coef: number[]; intercept: number }} params The object that contains the model parameters,
   * coef, and intercept that we need for our model.
   *
   * @returns {SGD} Returns the predictions.
   *
   * We use a LinearRegression in the example below because it provides
   * defaults for the SGD
   *
   * @example
   *
   * lr = new LinearRegression()
   * lr.importModel({coef : [1.2, 2.3], intercept: 10.0});
   * // lr model weights have been updated
   */

  importModel(params: { coef: number[]; intercept: number }): SGDRegressor {
    // Next steps: Need to update for possible 2D coef case, and 1D intercept case
    let myCoef = this.tf.tensor2d(
      params.coef,
      [params.coef.length, 1],
      'float32'
    )
    let myIntercept = this.tf.tensor1d([params.intercept], 'float32')
    this.initializeModel(myCoef, myIntercept, [myCoef, myIntercept])
    return this
  }

  /**
   * Similar to scikit-learn, this returns the object of configuration params for SGD
   * @returns {SGDRegressorParams} Returns an object of configuration params.
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

  getParams(): SGDRegressorParams {
    return {
      modelFitArgs: this.modelFitArgs,
      modelCompileArgs: this.modelCompileArgs,
      denseLayerArgs: this.denseLayerArgs,
      optimizerType: this.optimizerType,
      lossType: this.lossType
    }
  }

  /**
   * Similar to scikit-learn, this returns the object of configuration params for SGD
   * @returns {SGDRegressorParams} Returns an object of configuration params.
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

  setParams(params: SGDRegressorParams): SGDRegressor {
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
   * @param {Scikit2D} X The 2DTensor / 2D Array that you wish to run through
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

  public predict(X: Scikit2D): Tensor1D | Tensor2D {
    let XTwoD = convertToNumericTensor2D(X)
    if (this.model.layers.length === 0) {
      throw new RangeError('Need to call "fit" before "predict"')
    }
    const predictions = this.model.predict(XTwoD) as Tensor2D
    if (!this.isMultiOutput) {
      return predictions.reshape([-1]) as Tensor1D
    }
    return predictions as Tensor2D
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
   * lr.coef
   * // => tensor1d([[ 1.2, 3.3, 1.1, 0.2 ]])
   *
   * await lr.fit(X, [ [1,2], [3,4], [5,6] ]);
   * lr.coef
   * // => tensor2d([ [1.2, 3.3], [3.4, 5.6], [4.5, 6.7] ])

   */

  get coef(): Tensor1D | Tensor2D {
    const modelWeights = this.model.getWeights()
    if (modelWeights.length === 0) {
      return this.tf.tensor2d([])
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
   * lr.intercept
   * // => 4.5
   *
   *
   * lr = new LinearRegression()
   * await lr.fit(X, [ [1,2,3], [4,5,6] ]);
   * lr.intercept
   * // => tensor1d([1.2, 2.3])
   */
  get intercept(): number | Tensor1D {
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
