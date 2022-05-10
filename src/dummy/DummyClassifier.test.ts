import { DummyClassifier, setBackend, fromJSON } from '../index'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

describe('DummyClassifier', function () {
  it('Use DummyClassifier on simple example (mostFrequent)', function () {
    const clf = new DummyClassifier()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]
    const y = [10, 20, 20, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1]
    ]

    clf.fit(X, y)
    expect(clf.predict(predictX).arraySync()).toEqual([20, 20, 20])
  })
  it('Use DummyClassifier on simple example (constant)', function () {
    const clf = new DummyClassifier({ strategy: 'constant', constant: 10 })

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]
    const y = [10, 12, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1]
    ]

    clf.fit(X, y)
    expect(clf.predict(predictX).arraySync()).toEqual([10, 10, 10])
  })
  it('Use DummyClassifier on simple example (uniform)', function () {
    const scaler = new DummyClassifier({ strategy: 'uniform' })
    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]

    const y = [1, 2, 3]
    scaler.fit(X, y)

    expect(scaler.classes).toEqual([1, 2, 3])
  })
  it('should serialize DummyClassifier', async function () {
    const clf = new DummyClassifier()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]
    const y = [10, 20, 20, 30]
    const expectedResult = {
      name: 'DummyClassifier',
      EstimatorType: 'classifier',
      constant: 20,
      strategy: 'mostFrequent',
      classes: [10, 20, 30]
    }

    clf.fit(X, y)
    const clfSave = await clf.toObject()
    // We don't care what version of tf is saved on there
    delete clfSave.tf
    expect(expectedResult).toEqual(clfSave)
  })
  it('should load DummyClassifier', async function () {
    const clf = new DummyClassifier()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]
    const y = [10, 20, 20, 30]

    clf.fit(X, y)
    const clfSave = await clf.toJSON()
    const newClf = await fromJSON(clfSave)
    expect(clf).toEqual(newClf)
  })
})
