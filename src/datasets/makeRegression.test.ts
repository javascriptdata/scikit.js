import { makeLowRankMatrix, makeRegression, setBackend } from '../index'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

describe('makeRegression tests', () => {
  it('returns the right size', () => {
    let [X, y, model] = makeRegression({
      nSamples: 10,
      nFeatures: 20,
      nTargets: 5,
      coef: true
    })
    expect(X.shape).toEqual([10, 20])
    expect(y.shape).toEqual([10, 5])
    expect(model?.shape).toEqual([20, 5])
  })
  it('models the right size', () => {
    let [X, y, model] = makeRegression({
      nSamples: 10,
      nFeatures: 20,
      nTargets: 5,
      noise: 0,
      coef: true
    })
    expect(X.dot(model as tf.Tensor).dataSync()).toEqual(y.dataSync())
  })
  it('test low rank matrix', () => {
    let X = makeLowRankMatrix({
      nSamples: 10,
      nFeatures: 20
    })
    expect(X.shape).toEqual([10, 20])
  })
})
