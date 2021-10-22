import { assert } from 'chai'
import { DummyClassifier } from '../../dist'

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
    assert.deepEqual(clf.predict(predictX), [20, 20, 20])
  })
  it('Use DummyClassifier on simple example (constant)', function () {
    const clf = new DummyClassifier('constant', 10)

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
    assert.deepEqual(clf.predict(predictX), [10, 10, 10])
  })
  it('Use DummyClassifier on simple example (uniform)', function () {
    const scaler = new DummyClassifier('uniform')
    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10]
    ]

    const y = [1, 2, 3]
    scaler.fit(X, y)

    assert.deepEqual(scaler.$uniques, [1, 2, 3])
  })
})
