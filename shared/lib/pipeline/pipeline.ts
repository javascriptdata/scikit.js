/* eslint-disable @typescript-eslint/no-explicit-any */
import { assert } from '../types.utils'
import { Scikit1D, Scikit2D } from '../types'
import { Tensor2D } from '@tensorflow/tfjs-core'

/*
Next steps:
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
  constructor({ steps = [] }: PipelineParams = {}) {
    this.steps = steps
  }

  public async fit(X: Scikit2D, y: Scikit1D): Promise<Pipeline> {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      const [, transformer] = this.steps[i]
      assert(transformer.fitTransform, `${transformer} is not a Transformer`)
      XT = transformer.fitTransform(XT, y)
    }

    const [, predictor] = this.steps[this.steps.length - 1]
    await predictor.fit(XT, y)
    return this
  }

  public transform(X: Scikit2D): Tensor2D {
    let XT = X
    for (let i = 0; i < this.steps.length; i++) {
      const [, transformer] = this.steps[i]
      XT = (transformer as any).transform(XT)
    }

    return XT as Tensor2D
  }

  public fitTransform(X: Scikit2D, y: Scikit1D): Tensor2D {
    let XT = X
    for (let i = 0; i < this.steps.length; i++) {
      let [, transformer] = this.steps[i]
      XT = transformer.fitTransform(XT, y)
    }

    return XT as Tensor2D
  }

  public predict(X: Scikit2D) {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      let [, transformer] = this.steps[i]
      XT = (transformer as any).transform(XT)
    }

    const [, predictor] = this.steps[this.steps.length - 1]
    return (predictor as any).predict(XT)
  }

  public fitPredict(X: Scikit2D, y: Scikit1D) {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      let [, transformer] = this.steps[i]
      XT = transformer.fitTransform(XT, y)
    }

    const [, predictor] = this.steps[this.steps.length - 1]
    return (predictor as any).predict(XT, y)
  }
}
