import { assert } from 'chai'
import { DummyClassifier } from '../../dist'

describe('DummyClassifier', function () {
  it('Use DummyClassifier on simple example (mostFrequent)', function () {
    const clf = new DummyClassifier()

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10],
    ]
    const y = [10, 20, 20, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1],
    ]

    clf.fit(X, y)
    assert.deepEqual(clf.predict(predictX), [20, 20, 20])
  })
  it('Use DummyClassifier on simple example (constant)', function () {
    const clf = new DummyClassifier('constant', 10)

    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
    ]
    const y = [10, 12, 30]
    const predictX = [
      [1, 0],
      [1, 1],
      [1, 1],
    ]

    clf.fit(X, y)
    assert.deepEqual(clf.predict(predictX), [10, 10, 10])
  })
  it('Use DummyClassifier on simple example (uniform)', function () {
    const scaler = new DummyClassifier('uniform')
    const X = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
    ]

    const y = [1, 2, 3]
    scaler.fit(X, y)

    assert.deepEqual(scaler.$uniques, [1, 2, 3])
  })
  // it('InverseTransform with MaxAbsScaler', function () {
  //   const scaler = new MaxAbsScaler()
  //   scaler.fit([1, 5, 10, 10, 5]) // scaling factor is 10
  //   const resultTransform = scaler.transform([10, 5, 50, 100, 50])
  //   assert.deepEqual(resultTransform, [1, 0.5, 5, 10, 5])

  //   const resultInverse = scaler.inverseTransform([0.1, 0.5, 1, 1, 0.5])
  //   assert.deepEqual([1, 5, 10, 10, 5], resultInverse)
  // })
  // it('Index and columns are kept after transformation', function () {
  //   const data = [
  //     [-1, 2],
  //     [-0.5, 6],
  //     [0, 10],
  //     [1, 18],
  //   ]
  //   const df = new DataFrame(data, {
  //     index: [1, 2, 3, 4],
  //     columns: ['a', 'b'],
  //   })

  //   const scaler = new MaxAbsScaler()
  //   scaler.fit(df)
  //   const resultDf = scaler.transform(df) as DataFrame

  //   assert.deepEqual(resultDf.index, [1, 2, 3, 4])
  //   assert.deepEqual(resultDf.columns, ['a', 'b'])
  // })
  // it('Standardize values in a Series using a MaxAbsScaler', function () {
  //   const data = [5, 5, 10, -10, -5]
  //   const scaler = new MaxAbsScaler()
  //   const result = [0.5, 0.5, 1, -1, -0.5]

  //   scaler.fit(new Series(data))
  //   assert.deepEqual(
  //     (scaler.transform(new Series(data)) as Series).values,
  //     result
  //   )
  //   assert.deepEqual(scaler.transform([100, -1000]), [10, -100])
  // })
  // it('Handles pathological examples with constant features with MaxAbsScaler', function () {
  //   const data = [0, 0, 0, 0]
  //   const scaler = new MaxAbsScaler()
  //   scaler.fit(data)
  //   assert.deepEqual(scaler.transform([0, 0, 0, 0]), [0, 0, 0, 0])

  //   assert.deepEqual(scaler.transform([10, 10, -10, 10]), [10, 10, -10, 10])
  // })
  // it('Errors when you pass garbage input into a MaxAbsScaler', function () {
  //   const data = 4
  //   const scaler = new MaxAbsScaler()
  //   assert.throws(() => scaler.fit(data as any))
  // })
  // it('Gracefully handles Nan as inputs MaxAbsScaler', function () {
  //   const data = [4, 4, 'whoops', 4, -4]
  //   const scaler = new MaxAbsScaler()
  //   scaler.fit(data as any)
  //   assert.deepEqual(scaler.transform(data as number[]), [1, 1, NaN, 1, -1])
  // })
})
