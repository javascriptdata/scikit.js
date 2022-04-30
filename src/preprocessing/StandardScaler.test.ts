import { StandardScaler, setBackend } from '../index'
import * as dfd from 'danfojs-node'
import * as tf from '@tensorflow/tfjs-node'
setBackend(tf)

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
    expect(resultDf.values).toEqual(expected)
    expect(scaler.transform([[2, 2]]).arraySync()).toEqual([[3, 3]])
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
    expect(resultDf.values).toEqual(expected)
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

    expect(resultDf.arraySync()).toEqual(data)
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

    expect(scaler.transform(data).arraySync()).toEqual(expected)
    expect(scaler.transform([[2, 2]]).arraySync()).toEqual([[3, 3]])
  })

  it('StandardScaler works with constant column', function () {
    const data = [[1, 1, 1, 1, 1, 1, 1, 1]]

    const scaler = new StandardScaler()
    scaler.fit(data)
    const expected = [[0, 0, 0, 0, 0, 0, 0, 0]]

    expect(scaler.transform(data).arraySync()).toEqual(expected)
  })
  it('StandardScaler plays nice with Nan', function () {
    const scaler = new StandardScaler()
    scaler.fit([[1], ['NaN'], [1]] as any)
    expect(scaler.transform([[1, 1, 1]]).arraySync()).toEqual([[0, 0, 0]])
  })
  it('keeps track of variables', function () {
    let myDf = new dfd.DataFrame({ a: [1, 2, 3, 4], b: [5, 6, 7, 8] })
    let scaler = new StandardScaler()
    scaler.fit(myDf)
    expect(scaler.nSamplesSeen).toEqual(4)
    expect(scaler.nFeaturesIn).toEqual(2)
    expect(scaler.featureNamesIn).toEqual(['a', 'b'])
  })
  it('StandardScaler works with constant column, no centering', function () {
    const data = [[1, 1, 1, 1, 1, 1, 1, 1]]

    const scaler = new StandardScaler({ withMean: false })
    scaler.fit(data)
    const expected = [[1, 1, 1, 1, 1, 1, 1, 1]]

    expect(scaler.transform(data).arraySync()).toEqual(expected)
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

    expect(scaler.transform(data).arraySync()).toEqual(expected)
  })
})
