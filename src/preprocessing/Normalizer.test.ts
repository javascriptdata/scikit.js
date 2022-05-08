import { Normalizer } from '../index'
import * as dfd from 'danfojs-node'
import { arrayEqual } from '../utils'

describe('Normalizer', function () {
  it('Standardize values in a DataFrame using a Normalizer (l1 case)', function () {
    const data = [
      [-1, 1],
      [-6, 6],
      [0, 10],
      [10, 20]
    ]
    const scaler = new Normalizer({ norm: 'l1' })

    const expected = [
      [-0.5, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [0.33, 0.66]
    ]
    const transformedData = [[0.5, 0.5]]

    scaler.fit(new dfd.DataFrame(data))
    const resultDf = new dfd.DataFrame(
      scaler.transform(new dfd.DataFrame(data))
    )
    expect(arrayEqual(resultDf.values, expected, 0.1)).toBe(true)
    expect(scaler.transform([[2, 2]]).arraySync()).toEqual(transformedData)
  })
  it('fitTransform using a Normalizer (l2 case)', function () {
    const data = [
      [-1, 2],
      [-3, 4],
      [0, 10]
    ]
    const scaler = new Normalizer()
    const resultDf = scaler.fitTransform(data)

    const expected = [
      [-1 / Math.sqrt(5), 2 / Math.sqrt(5)],
      [-0.6, 0.8],
      [0, 1]
    ]
    expect(arrayEqual(resultDf.arraySync(), expected, 0.1)).toBe(true)
  })
  it('fitTransform using a Normalizer (max case)', function () {
    const data = [
      [-1, 2],
      [-3, 4],
      [0, 10]
    ]
    const scaler = new Normalizer({ norm: 'max' })
    const resultDf = scaler.fitTransform(data)

    const expected = [
      [-0.5, 1],
      [-0.75, 1],
      [0, 1]
    ]
    expect(arrayEqual(resultDf.arraySync(), expected, 0.1)).toBe(true)
  })
  it('fitTransform using a Normalizer (l1 case)', function () {
    const data = [
      [-1, 2],
      [-3, 4],
      [0, 0]
    ]
    const scaler = new Normalizer({ norm: 'l1' })
    const resultDf = scaler.fitTransform(data)
    const expected = [
      [-1 / 3, 2 / 3],
      [-3 / 7, 4 / 7],
      [0, 0]
    ]
    expect(arrayEqual(resultDf.arraySync(), expected, 0.1)).toBe(true)
  })
})
