import { Scikit1D, Scikit2D } from '../types'
import { tf } from '../shared/globals'
import { ClassifierMixin } from '../mixins'
import { tensor1d, Tensor1D, Tensor2D } from '@tensorflow/tfjs-core'
import { LabelEncoder } from '../preprocessing/labelEncoder'
/*
  Next steps:
  0. Write validation code to check Estimator inputs
  1. nFeaturesIn, featureNamesIn
  2. Copy most of the code for the VotingClassifier
*/

export interface VotingClassifierParams {
  /** List of name, estimator pairs. Example
   * `[['lr', new LinearRegression()], ['dt', new DecisionTree()]]`
   */
  estimators?: Array<[string, any]>

  /** The weights for the estimators. If not present, then there is a uniform weighting. */
  weights?: number[]

  /** If ‘hard’, uses predicted class labels for majority rule voting.
   * Else if ‘soft’, predicts the class label based on the argmax of the
   *  sums of the predicted probabilities, which is recommended for an
   * ensemble of well-calibrated classifiers.
   */
  voting?: 'hard' | 'soft'
}

/**
 * A voting regressor is an ensemble meta-estimator that fits several base
 * regressors, each on the whole dataset. Then it averages the individual
 * predictions to form a final prediction.
 *
 * @example
 * ```js
 * import { VotingClassifier, DummyClassifier, LogisticRegression } from 'scikitjs'
 *
 * const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1],
      [4, 4]
    ]
    const y = [0, 0, 1, 1, 1]
    const voter = new VotingClassifier({
      estimators: [
        ['dt', new DummyClassifier()],
        ['dt', new DummyClassifier()],
        ['lr', new LogisticRegression({ penalty: 'none' })]
      ]
    })

    await voter.fit(X, y)
    assert.deepEqual(voter.predict(X).arraySync(), [1, 1, 1, 1, 1])
 * ```
 */
export class VotingClassifier extends ClassifierMixin {
  estimators: Array<[string, any]>
  weights?: number[]
  le: any
  name = 'VotingClassifier'

  constructor({
    estimators = [],
    weights = undefined,
    voting = 'hard'
  }: VotingClassifierParams = {}) {
    super()
    this.estimators = estimators
    this.weights = weights
    this.voting = voting
    this.le = new LabelEncoder()
  }

  public async fit(X: Scikit2D, y: Scikit1D): Promise<VotingClassifier> {
    let newY = this.le.fitTransform(y)
    for (let i = 0; i < this.estimators?.length; i++) {
      let [_, curEstimator] = this.estimators[i]
      await curEstimator.fit(X, newY)
    }
    return this
  }

  public predictProba(X: Scikit2D): Tensor1D {
    let responses = []
    let numEstimators = this.estimators.length
    const weights =
      this.weights || Array(numEstimators).fill(1 / numEstimators)
    for (let i = 0; i < numEstimators; i++) {
      let [_, curEstimator] = this.estimators[i]
      let curWeight = weights[i]
      responses.push(curEstimator.predictProba(X).mul(curWeight))
    }

    return tf.addN(responses)
  }

  // only hard case
  public predict(X: Scikit2D): Tensor1D {
    let responses = []
    let numEstimators = this.estimators.length
    const weights =
      this.weights || Array(numEstimators).fill(1 / numEstimators)

    if (this.voting === 'hard') {
      for (let i = 0; i < numEstimators; i++) {
        let [_, curEstimator] = this.estimators[i]
        let curWeight = weights[i]
        let predictions = curEstimator.predict(X).toInt()
        let oneHot = tf.oneHot(predictions, this.le.classes.length)
        responses.push(oneHot.mul(curWeight))
      }
      return tensor1d(this.le.inverseTransform(tf.addN(responses).argMax(1)))
    } else {
      for (let i = 0; i < numEstimators; i++) {
        let [_, curEstimator] = this.estimators[i]
        let curWeight = weights[i]
        let predictions = curEstimator.predictProba(X)
        responses.push(predictions.mul(curWeight))
      }
      return tensor1d(this.le.inverseTransform(tf.addN(responses).argMax(1)))
    }
  }

  public transform(X: Scikit2D): Array<Tensor1D> | Array<Tensor2D> {
    let responses = []
    let numEstimators = this.estimators.length

    if (this.voting === 'hard') {
      for (let i = 0; i < numEstimators; i++) {
        let [_, curEstimator] = this.estimators[i]
        responses.push(curEstimator.predict(X))
      }
      return responses
    } else {
      for (let i = 0; i < numEstimators; i++) {
        let [_, curEstimator] = this.estimators[i]
        responses.push(curEstimator.predictProba(X))
      }
      return responses
    }
  }

  public async fitTransform(
    X: Scikit2D,
    y: Scikit1D
  ): Promise<Array<Tensor1D> | Array<Tensor2D>> {
    return (await this.fit(X, y)).transform(X)
  }
}

export function makeVotingClassifier(...args: any[]) {
  let estimators: Array<[string, any]> = []
  for (let i = 0; i < args.length; i++) {
    // eslint-disable-next-line prefer-rest-params
    let cur = args[i]
    estimators.push([cur.name, cur])
  }
  return new VotingClassifier({ estimators })
}
