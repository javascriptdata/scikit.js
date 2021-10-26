import { assert } from 'chai'
import { RobustScaler } from '../../../dist'
import { DataFrame } from 'danfojs-node'
import { arrayEqual } from '../../utils'

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

    scaler.fit(new DataFrame(X))
    const resultDf = new DataFrame(scaler.transform(new DataFrame(X)))
    assert.isTrue(arrayEqual(resultDf.values, expected, 0.1))
  })
})
