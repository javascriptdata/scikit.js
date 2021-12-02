import { assert } from 'chai'
import { StandardScaler } from './standard.scaler'
import { dfd } from '../../globals'
import { describe, it } from 'mocha'

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
  it('keeps track of variables', function () {
    let myDf = new dfd.DataFrame({ a: [1, 2, 3, 4], b: [5, 6, 7, 8] })
    let scaler = new StandardScaler()
    scaler.fit(myDf)
    assert.deepEqual(scaler.nSamplesSeen, 4)
    assert.deepEqual(scaler.nFeaturesIn, 2)
    assert.deepEqual(scaler.featureNamesIn, ['a', 'b'])
  })
  it('StandardScaler works with constant column, no centering', function () {
    const data = [[1, 1, 1, 1, 1, 1, 1, 1]]

    const scaler = new StandardScaler({ withMean: false })
    scaler.fit(data)
    const expected = [[1, 1, 1, 1, 1, 1, 1, 1]]

    assert.deepEqual(scaler.transform(data).arraySync(), expected)
  })
  it('StandardScaler works for Array no std', function () {
    const data = [
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1]
    ]

    const scaler = new StandardScaler({ withStd: false })
    scaler.fit(data)
    const expected = [
      [-0.5, -0.5],
      [-0.5, -0.5],
      [0.5, 0.5],
      [0.5, 0.5]
    ]

    assert.deepEqual(scaler.transform(data).arraySync(), expected)
  })
})
