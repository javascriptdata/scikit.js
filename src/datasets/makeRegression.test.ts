import { makeLowRankMatrix, makeRegression } from './makeRegression'
import { Tensor } from '@tensorflow/tfjs-core'
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
    expect(X.dot(model as Tensor).dataSync()).toEqual(y.dataSync())
  })
  it('test low rank matrix', () => {
    let X = makeLowRankMatrix({
      nSamples: 10,
      nFeatures: 20
    })
    expect(X.shape).toEqual([10, 20])
  })
})
