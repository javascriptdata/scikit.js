import { Scikit1D, Scikit2D } from '../types'
import { tf } from '../../globals'
import { Tensor2D } from '@tensorflow/tfjs-core'
import { RegressorMixin } from '../mixins'
/*
  Next steps:
  1. Add doc strings to interface above
  2. Add example above the VotingRegressor yourself
  3. nFeaturesIn, featureNamesIn
  3. Copy most of the code for the VotingClassifier
*/

export interface VotingRegressorParams {
  /** List of name, estimator pairs. Example
   * `[['lr', new LinearRegression()], ['dt', new DecisionTree()]]`
   */
  estimators?: Array<[string, any]>

  /** The weights for the estimators. If not present, then there is a uniform weighting. */
  weights?: number[]
}

/**
 * A voting regressor is an ensemble meta-estimator that fits several base
 * regressors, each on the whole dataset. Then it averages the individual
 * predictions to form a final prediction.
 *
 * @example
 * ```js
 * import { VotingRegressor, DecisionTreeRegressor, LinearRegression } from 'scikitjs'
 *
 * const X = [
      [2, 2],
      [2, 3],
      [5, 4],
      [1, 0]
    ]
    const y = [5, 3, 4, 1.5]
    const voter = new VotingRegressor({
      estimators: [
        ['dt', new DecisionTreeRegressor()],
        ['lr', new LinearRegression({ fitIntercept: false })]
      ]
    })

    await pipeline.fit(X, y)
 * ```
 */
export class VotingRegressor extends RegressorMixin {
  estimators: Array<[string, any]>
  weights?: number[]

  name = 'VotingRegressor'

  constructor({
    estimators = [],
    weights = undefined
  }: VotingRegressorParams = {}) {
    super()
    this.estimators = estimators
    this.weights = weights
  }

  public async fit(X: Scikit2D, y: Scikit1D): Promise<VotingRegressor> {
    for (let i = 0; i < this.estimators?.length; i++) {
      let [_, curEstimator] = this.estimators[i]
      await curEstimator.fit(X, y)
    }
    return this
  }

  public predict(X: Scikit2D): Tensor2D {
    let responses = []
    let numEstimators = this.estimators.length
    for (let i = 0; i < numEstimators; i++) {
      let [_, curEstimator] = this.estimators[i]
      responses.push(curEstimator.predict(X))
    }
    const weights =
      this.weights || Array(numEstimators).fill(1 / numEstimators)
    for (let i = 0; i < weights.length; i++) {
      let curWeight = weights[i]
      responses[i] = responses[i].mul(curWeight)
    }
    return tf.addN(responses)
  }

  public async fitPredict(X: Scikit2D, y: Scikit1D) {
    return (await this.fit(X, y)).predict(X)
  }
}
