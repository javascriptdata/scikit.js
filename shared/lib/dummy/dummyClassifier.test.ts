import { assert } from 'chai'
import { DummyClassifier } from './dummyClassifier'
import { describe, it } from 'mocha'

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
    assert.deepEqual(clf.predict(predictX).arraySync(), [20, 20, 20])
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
    assert.deepEqual(clf.predict(predictX).arraySync(), [10, 10, 10])
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

    assert.deepEqual(scaler.classes, [1, 2, 3])
  })
  it('should serialize DummyClassifier', function () {
    const clf = new DummyClassifier()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]
    const y = [10, 20, 20, 30]
    const expectedResult = { "name":"dummyclassifier", "EstimatorType":"classifier", "constant":20, "strategy":"mostFrequent", "classes":[10, 20, 30] }

    clf.fit(X, y)
    const clfSave = clf.toJson()
    assert.deepEqual(expectedResult, JSON.parse(clfSave))
  })
  it('should load DummyClassifier', function () {
    const clf = new DummyClassifier()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]
    const y = [10, 20, 20, 30]

    clf.fit(X, y)
    const clfSave = clf.toJson()
    const newClf = new DummyClassifier().fromJson(clfSave)
    assert.deepEqual(clf, newClf)
  })
})
