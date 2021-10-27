/* eslint-disable @typescript-eslint/no-explicit-any */
import { assert } from 'console'
import { Scikit1D, Scikit2D } from 'types'

export type Bunch = [string, any]

export default class Pipeline {
  steps: Array<Bunch>
  constructor(steps: Array<Bunch>) {
    this.steps = steps
  }

  async fit(X: Scikit2D, y: Scikit1D) {
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

  transform(X: Scikit2D) {
    let XT = X
    for (let i = 0; i < this.steps.length; i++) {
      const [, transformer] = this.steps[i]
      XT = (transformer as any).transform(XT)
    }

    return XT
  }

  fitTransform(X: Scikit2D, y: Scikit1D) {
    let XT = X
    for (let i = 0; i < this.steps.length; i++) {
      let [, transformer] = this.steps[i]
      XT = transformer.fitTransform(XT, y)
    }

    return XT
  }

  predict(X: Scikit2D) {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      let [, transformer] = this.steps[i]
      XT = (transformer as any).transform(XT)
    }

    const [, predictor] = this.steps[this.steps.length - 1]
    return (predictor as any).predict(XT)
  }

  fitPredict(X: Scikit2D, y: Scikit1D) {
    let XT = X
    for (let i = 0; i < this.steps.length - 1; i++) {
      let [, transformer] = this.steps[i]
      XT = transformer.fitTransform(XT, y)
    }

    const [, predictor] = this.steps[this.steps.length - 1]
    return (predictor as any).predict(XT, y)
  }
}
