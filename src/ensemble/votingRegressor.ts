import { Scikit1D, Scikit2D } from '../types'
import { tf } from '../shared/globals'
import { RegressorMixin } from '../mixins'
import { Tensor1D } from '@tensorflow/tfjs-core'
import { fromJson, toJson } from './serializeEnsemble'
/*
  Next steps:
  0. Write validation code to check Estimator inputs
  1. nFeaturesIn, featureNamesIn
  2. Copy most of the code for the VotingClassifier
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

    await voter.fit(X, y)
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

  public predict(X: Scikit2D): Tensor1D {
    let responses = []
    let numEstimators = this.estimators.length
    const weights =
      this.weights || Array(numEstimators).fill(1 / numEstimators)
    for (let i = 0; i < numEstimators; i++) {
      let [_, curEstimator] = this.estimators[i]
      let curWeight = weights[i]
      responses.push(curEstimator.predict(X).mul(curWeight))
    }

    return tf.addN(responses)
  }

  public transform(X: Scikit2D): Array<Tensor1D> {
    let responses = []
    let numEstimators = this.estimators.length
    for (let i = 0; i < numEstimators; i++) {
      let [_, curEstimator] = this.estimators[i]
      responses.push(curEstimator.predict(X))
    }
    return responses
  }

  public async fitTransform(X: Scikit2D, y: Scikit1D) {
    return (await this.fit(X, y)).transform(X)
  }

  public fromJson(model: string) {
    return fromJson(this, model) as this
  }

  public async toJson(): Promise<string> {
    const classJson = JSON.parse(super.toJson() as string)
    return toJson(this, classJson)
  }
}

/**
 *
 * Helper function for make a VotingRegressor. Just pass your Estimators as function arguments.
 *
 * @example
 * ```typescript
 * import {makeVotingRegressor, DummyRegressor, LinearRegression} from 'scikitjs'
 *  const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1]
    ]
    const y = [3, 3, 4, 4]
    const voter = makeVotingRegressor(
      new DummyRegressor(),
      new LinearRegression({ fitIntercept: true })
    )

    await voter.fit(X, y)
    ```
 */
export function makeVotingRegressor(...args: any[]) {
  let estimators: Array<[string, any]> = []
  for (let i = 0; i < args.length; i++) {
    // eslint-disable-next-line prefer-rest-params
    let cur = args[i]
    estimators.push([cur.name, cur])
  }
  return new VotingRegressor({ estimators })
}
