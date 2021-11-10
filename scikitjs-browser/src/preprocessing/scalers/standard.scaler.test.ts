import { assert } from 'chai'
import { StandardScaler } from '../../../dist'
import { dfd } from '../../globals'

describe('StandardScaler', function () {
  it('StandardScaler works for DataFrame', function () {
    const data = [
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1]
    ]

    const scaler = new StandardScaler()
    scaler.fit(new dfd.DataFrame(data))

    const expected = [
      [-1, -1],
      [-1, -1],
      [1, 1],
      [1, 1]
    ]
    const resultDf = new dfd.DataFrame(
      scaler.transform(new dfd.DataFrame(data))
    )
    assert.deepEqual(resultDf.values, expected)
    assert.deepEqual(scaler.transform([[2, 2]]).arraySync(), [[3, 3]])
  })
  it('fitTransform works for StandardScaler', function () {
    const data = [
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1]
    ]

    const scaler = new StandardScaler()
    const resultDf = new dfd.DataFrame(
      scaler.fitTransform(new dfd.DataFrame(data))
    )

    const expected = [
      [-1, -1],
      [-1, -1],
      [1, 1],
      [1, 1]
    ]
    assert.deepEqual(resultDf.values, expected)
  })
  it('inverseTransform works for StandardScaler', function () {
    const data = [
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1]
    ]

    const scaler = new StandardScaler()
    scaler.fit(data)
    const resultDf = scaler.inverseTransform([
      [-1, -1],
      [-1, -1],
      [1, 1],
      [1, 1]
    ])

    assert.deepEqual(resultDf.arraySync(), data)
  })
  it('StandardScaler works for Array', function () {
    const data = [
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1]
    ]

    const scaler = new StandardScaler()
    scaler.fit(data)
    const expected = [
      [-1, -1],
      [-1, -1],
      [1, 1],
      [1, 1]
    ]

    assert.deepEqual(scaler.transform(data).arraySync(), expected)
    assert.deepEqual(scaler.transform([[2, 2]]).arraySync(), [[3, 3]])
  })

  it('StandardScaler works with constant column', function () {
    const data = [[1, 1, 1, 1, 1, 1, 1, 1]]

    const scaler = new StandardScaler()
    scaler.fit(data)
    const expected = [[0, 0, 0, 0, 0, 0, 0, 0]]

    assert.deepEqual(scaler.transform(data).arraySync(), expected)
  })
  it('StandardScaler plays nice with Nan', function () {
    const scaler = new StandardScaler()
    scaler.fit([[1], ['NaN'], [1]] as any)
    assert.deepEqual(scaler.transform([[1, 1, 1]]).arraySync(), [[0, 0, 0]])
  })
})
