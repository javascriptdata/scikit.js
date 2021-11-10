import { assert } from 'chai'
import { MaxAbsScaler } from '../../../dist'
import { dfd } from '../../globals'
import { tensor2d } from '@tensorflow/tfjs-core'

describe('MaxAbsScaler', function () {
  it('Standardize values in a DataFrame using a MaxAbsScaler', function () {
    const scaler = new MaxAbsScaler()

    const data = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]

    const expected = [
      [-1, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [1, 1]
    ]

    scaler.fit(new dfd.DataFrame(data))
    const resultDf = new dfd.DataFrame(
      scaler.transform(new dfd.DataFrame(data))
    )
    assert.deepEqual(resultDf.values, expected)
    assert.deepEqual(scaler.transform([[2, 5]]).arraySync(), [[2, 0.5]])
  })
  it('fitTransform using a MaxAbsScaler', function () {
    const scaler = new MaxAbsScaler()
    const data = [
      [-1, 5],
      [-0.5, 5],
      [0, 10],
      [1, 10]
    ]

    const expected = [
      [-1, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [1, 1]
    ]
    const resultDf = new dfd.DataFrame(
      scaler.fitTransform(new dfd.DataFrame(data))
    )

    assert.deepEqual(resultDf.values, expected)
  })
  it('InverseTransform with MaxAbsScaler', function () {
    const scaler = new MaxAbsScaler()
    // const first = [1, 5, 10, 10, 5]
    // const second = [10, 5, 50, 100, 50]
    // const third = [1, 0.5, 5, 10, 5]
    scaler.fit(tensor2d([1, 5, 10, 10, 5], [5, 1])) // scaling factor is 10
    const resultTransform = scaler.transform(
      tensor2d([10, 5, 50, 100, 50], [5, 1])
    )
    assert.deepEqual(resultTransform.arraySync().flat(), [1, 0.5, 5, 10, 5])

    const resultInverse = scaler.inverseTransform(
      tensor2d([0.1, 0.5, 1, 1, 0.5], [5, 1])
    )
    assert.deepEqual([1, 5, 10, 10, 5], resultInverse.arraySync().flat())
  })
  it('Handles pathological examples with constant features with MaxAbsScaler', function () {
    const data = [[0, 0, 0, 0]]
    const scaler = new MaxAbsScaler()
    scaler.fit(data)
    assert.deepEqual(scaler.transform([[0, 0, 0, 0]]).arraySync(), [
      [0, 0, 0, 0]
    ])

    assert.deepEqual(scaler.transform([[10, 10, -10, 10]]).arraySync(), [
      [10, 10, -10, 10]
    ])
  })
  it('Errors when you pass garbage input into a MaxAbsScaler', function () {
    const data = 4
    const scaler = new MaxAbsScaler()
    assert.throws(() => scaler.fit(data as any))
  })
  it('Gracefully handles Nan as inputs MaxAbsScaler', function () {
    const data = tensor2d([4, 4, 'whoops', 4, -4] as any, [5, 1])
    const scaler = new MaxAbsScaler()
    scaler.fit(data as any)
    assert.deepEqual(scaler.transform(data as any).arraySync(), [
      [1],
      [1],
      [NaN],
      [1],
      [-1]
    ])
  })
})
