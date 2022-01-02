import { RobustScaler } from './robustScaler'
import { dfd } from '../shared/globals'
import { arrayEqual } from '../utils'


describe('RobustScaler', function () {
  it('Standardize values in a DataFrame using a RobustScaler', function () {
    const X = [
      [1, -2, 2],
      [-2, 1, 3],
      [4, 1, -2]
    ]

    const scaler = new RobustScaler()

    const expected = [
      [0, -2, 0],
      [-1, 0, 0.4],
      [1, 0, -1.6]
    ]

    scaler.fit(new dfd.DataFrame(X))
    const resultDf = new dfd.DataFrame(scaler.transform(new dfd.DataFrame(X)))
    expect(arrayEqual(resultDf.values, expected, 0.1)).toBe(true)
  })
  it('Standardize values in a DataFrame using a RobustScaler quantileRange', function () {
    const X = [
      [1, -2, 2],
      [-2, 1, 3],
      [4, 1, -2]
    ]

    const scaler = new RobustScaler({ quantileRange: [0, 100] })

    const expected = [
      [0, -1, 0],
      [-0.5, 0, 0.2],
      [0.5, 0, -0.8]
    ]

    scaler.fit(new dfd.DataFrame(X))
    const resultDf = new dfd.DataFrame(scaler.transform(new dfd.DataFrame(X)))
    expect(arrayEqual(resultDf.values, expected, 0.1)).toBe(true)
  })
  it('Standardize values in a DataFrame using a RobustScaler quantileRange with no scaling', function () {
    const X = [
      [1, -2, 2],
      [-2, 1, 3],
      [4, 1, -2]
    ]

    const scaler = new RobustScaler({
      quantileRange: [0, 100],
      withScaling: false
    })

    const expected = [
      [0, -3, 0],
      [-3, 0, 1],
      [3, 0, -4]
    ]

    const resultDf = scaler.fitTransform(X)

    expect(arrayEqual(resultDf.arraySync(), expected, 0.1)).toBe(true)
  })
})
