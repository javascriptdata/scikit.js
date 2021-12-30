/* eslint-disable @typescript-eslint/no-explicit-any */
import { assert } from '../typesUtils'
import { Scikit1D, Scikit2D } from '../types'
import { Tensor2D } from '@tensorflow/tfjs-core'

/*
Next steps:
0. Implement nFeaturesIn, and featureNamesIn
1. Implement attribute classes
2. Pass next 5 scikit-learn tests
*/
export interface PipelineParams {
  steps?: Array<[string, any]>
}

/** Construct a pipeline of transformations, with the final one being an estimator.
 * Usually this is used to perform some cleaning of the data in the early stages of the pipeline
 * (ie. StandardScaling, or SimpleImputer), and then ending with the fitted estimator.
 *
 * <!-- prettier-ignore-start -->
 * ```js
 * import { Pipeline } from 'scikitjs'
 *
 * const X = [
      [2, 2], // [1, .5]
      [2, NaN], // [1, 0]
      [NaN, 4], // [0, 1]
      [1, 0] // [.5, 0]
    ]
    const y = [5, 3, 4, 1.5]
    const pipeline = new Pipeline({
      steps: [
        [
          'simpleImputer',
          new SimpleImputer({ strategy: 'constant', fillValue: 0 })
        ],
        ['minmax', new MinMaxScaler()],
        ['lr', new LinearRegression({ fitIntercept: false })]
      ]
    })

    await pipeline.fit(X, y)
 * ```
 */
export class Pipeline {
  steps: Array<[string, any]>

  /** Useful for pipelines and column transformers to have a default name for transforms */
  name = 'pipeline'

  constructor({ steps = [] }: PipelineParams = {}) {
    this.steps = steps
    this.validateSteps(this.steps)
  }

  /** Checks if the input is a Transformer or the string "passthrough" */
  isTransformer(possibleTransformer: any) {
    if (possibleTransformer === 'passthrough') {
      return true
    }
    if (
      typeof possibleTransformer.fit === 'function' &&
      typeof possibleTransformer.transform === 'function' &&
      typeof possibleTransformer.fitTransform === 'function'
    ) {
      return true
    }
    return false
  }

  /** Checks if the input is an Estimator or the string "passthrough" */
  isEstimator(possibleTransformer: any) {
    if (possibleTransformer === 'passthrough') {
      return true
    }
    if (typeof possibleTransformer.fit === 'function') {
      return true
    }
    return false
  }

  /** Checks if the steps are valid. Each of the elements in the array (except for the last)
   * must be a Transformer. That means they need a "fit" and "transform" method. The only special case
   * is the string "passthrough" which leaves the input untouched. The sklearn pipeline uses that feature
   * a lot when it grid searches through everything.
   *
   * I call validateSteps in the constructor as well as every call to fit/predict. In the case of grid search
   * the steps can be changed at runtime and so you need to check on every call if your value for steps is still
   * valid
   */
  validateSteps(steps: Array<[string, any]>) {
    assert(Array.isArray(steps), `steps is not an array. It is ${steps}`)
    if (steps.length === 0) {
      // Empty array is valid
      return
    }
    for (let i = 0; i < steps.length - 1; i++) {
      const step = steps[i]
      assert(
        Array.isArray(step),
        `A single step in your pipeline must be an array containing a string as the first argument, and the transformer in the second. Something akin to ['minmaxscaler', new MinMaxScaler()]. Instead it is ${step}`
      )
      assert(
        this.isTransformer(step[1]),
        `The ${i}th step in your pipeline isn't an array containing a name and a Transformer. Instead it is ${steps[i]}.`
      )
    }
    let lastEstimator = steps[steps.length - 1]
    assert(
      Array.isArray(lastEstimator),
      `The last element in your pipeline must be a 2-element array that contains a string as the first argument, and an estimator as the second. Instead it is ${lastEstimator}`
    )
    assert(
      this.isEstimator(lastEstimator[1]),
      `The last element in your pipeline should be an Estimator. Instead it is ${lastEstimator}`
    )
  }

  transformExceptLast(X: Scikit2D) {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [, transformer] = this.steps[i]
      if (transformer === 'passthrough') {
        continue
      }
      XT = transformer.transform(XT)
    }
    return XT
  }

  fitTransformExceptLast(X: Scikit2D) {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [, transformer] = this.steps[i]
      if (transformer === 'passthrough') {
        continue
      }
      XT = transformer.fitTransform(XT)
    }
    return XT
  }

  getLastEstimator() {
    if (this.steps.length === 0) {
      return null
    }
    return this.steps[this.steps.length - 1][1]
  }

  assertEstimatorHasFunction(estimator: any, funcName: string) {
    assert(
      estimator !== null,
      `Your final Estimator is null and therefore you can't call ${funcName}`
    )
    assert(
      typeof estimator[funcName] === 'function',
      `Estimator ${estimator} doesn't implement the function ${funcName}`
    )
  }

  public async fit(X: Scikit2D, y: Scikit1D): Promise<Pipeline> {
    this.validateSteps(this.steps)
    const lastEstimator = this.getLastEstimator()
    this.assertEstimatorHasFunction(lastEstimator, 'fit')

    let XT = this.fitTransformExceptLast(X)
    await lastEstimator.fit(XT, y)
    return this
  }

  public transform(X: Scikit2D): Tensor2D {
    this.validateSteps(this.steps)
    const lastEstimator = this.getLastEstimator()
    this.assertEstimatorHasFunction(lastEstimator, 'transform')

    let XT = this.transformExceptLast(X)
    return lastEstimator.transform(XT) as Tensor2D
  }

  public fitTransform(X: Scikit2D, y: Scikit1D): Tensor2D {
    this.validateSteps(this.steps)
    const lastEstimator = this.getLastEstimator()
    this.assertEstimatorHasFunction(lastEstimator, 'fitTransform')

    let XT = this.fitTransformExceptLast(X)
    return lastEstimator.fitTransform(XT) as Tensor2D
  }

  public predict(X: Scikit2D) {
    this.validateSteps(this.steps)
    const lastEstimator = this.getLastEstimator()
    this.assertEstimatorHasFunction(lastEstimator, 'predict')

    let XT = this.transformExceptLast(X)
    return lastEstimator.predict(XT)
  }

  public async fitPredict(X: Scikit2D, y: Scikit1D) {
    this.validateSteps(this.steps)
    const lastEstimator = this.getLastEstimator()
    this.assertEstimatorHasFunction(lastEstimator, 'fitPredict')

    let XT = this.fitTransformExceptLast(X)
    return await lastEstimator.fitPredict(XT, y)
  }
}

export function makePipeline(...args: any[]) {
  let pipelineSteps: Array<[string, any]> = []
  for (let i = 0; i < args.length; i++) {
    // eslint-disable-next-line prefer-rest-params
    let cur = args[i]
    pipelineSteps.push([cur.name, cur])
  }
  return new Pipeline({ steps: pipelineSteps })
}
