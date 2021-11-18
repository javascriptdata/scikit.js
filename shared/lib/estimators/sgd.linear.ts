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
  Tensor,
  Tensor1D,
  Tensor2D,
  tensor1d,
  tensor2d,
  losses
} from '@tensorflow/tfjs-core'
import {
  layers,
  sequential,
  Sequential,
  ModelFitArgs,
  ModelCompileArgs
} from '@tensorflow/tfjs-layers'
import { DenseLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/core'
import {
  convertToNumericTensor1D_2D,
  convertToNumericTensor2D
} from '../utils'
import { Scikit2D, ScikitVecOrMatrix } from '../types'
import { PredictorMixin } from '../mixins'
import OneHotEncoder from '../preprocessing/encoders/one.hot.encoder'
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

  /**
   * This class specifies that we are building a linear model that uses SGD. But there still is the
   * question, "Is this model performing classification or regression"? This argument answers that
   * definitely. It's a boolean that is true when the model aims to perform classification
   */

  isClassification?: boolean
}

export class SGD extends PredictorMixin {
  model: Sequential
  modelFitArgs: ModelFitArgs
  modelCompileArgs: ModelCompileArgs
  denseLayerArgs: DenseLayerArgs

  /* For Classification */
  isClassification: boolean
  oneHot: OneHotEncoder

  constructor({
    modelFitArgs,
    modelCompileArgs,
    denseLayerArgs,
    isClassification
  }: SGDParams) {
    super()
    this.model = sequential()
    this.modelFitArgs = modelFitArgs
    this.modelCompileArgs = modelCompileArgs
    this.denseLayerArgs = denseLayerArgs
    this.isClassification = Boolean(isClassification)
    // TODO: Implement "drop" mechanics for OneHotEncoder
    // There is a possibility to do a drop => if_binary which would
    // squash down on the number of variables that we'd have to learn
    this.oneHot = new OneHotEncoder()
  }

  initializeModelForClassification(y: Tensor1D | Tensor2D): Tensor2D {
    let yToInt = y.toInt()
    // This covers the case of a dependent variable that is already one hot encoded.
    // There are other cases where you do "multi-variable output which isn't one hot encoded"
    // Like say you were predicting which diseases a person could have (hasCancer, hasMeningitis, etc)
    // Then you would have to run a sigmoid on each independent variable
    if (yToInt.shape.length === 2) {
      this.modelCompileArgs.loss = losses.softmaxCrossEntropy
      return yToInt as Tensor2D
    } else {
      const yTwoD = y.reshape([-1, 1])
      const yTwoDOneHotEncoded = this.oneHot.fitTransform(yTwoD)
      if (this.oneHot.categories[0].length > 2) {
        this.modelCompileArgs.loss = losses.softmaxCrossEntropy
      } else {
        this.modelCompileArgs.loss = losses.sigmoidCrossEntropy
      }
      return yTwoDOneHotEncoded
    }
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
    const model = sequential()
    model.add(
      layers.dense({ inputShape: [X.shape[1]], ...this.denseLayerArgs })
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

  async fit(X: Scikit2D, y: ScikitVecOrMatrix): Promise<SGD> {
    let XTwoD = convertToNumericTensor2D(X)
    let yOneD = convertToNumericTensor1D_2D(y)

    if (this.isClassification) {
      yOneD = this.initializeModelForClassification(yOneD)
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

  importModel(params: { coef: number[]; intercept: number }): SGD {
    // TODO: Need to update for possible 2D coef case, and 1D intercept case
    let myCoef = tensor2d(params.coef, [params.coef.length, 1], 'float32')
    let myIntercept = tensor1d([params.intercept], 'float32')
    this.initializeModel(myCoef, myIntercept, [myCoef, myIntercept])
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
      isClassification: Boolean(this.isClassification)
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

  predict(X: Scikit2D): Tensor1D | Tensor2D {
    let XTwoD = convertToNumericTensor2D(X)
    if (this.model.layers.length === 0) {
      throw new RangeError('Need to call "fit" before "predict"')
    }
    if (this.isClassification) {
      let yLabels = this.model.predict(XTwoD) as Tensor2D
      return tensor1d(this.oneHot.inverseTransform(yLabels)).reshape([-1, 1])
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
