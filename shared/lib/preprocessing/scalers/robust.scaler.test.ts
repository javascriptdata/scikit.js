import { assert } from 'chai'
import { RobustScaler } from './robust.scaler'
import { dfd } from '../../../globals'
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

    scaler.fit(new dfd.DataFrame(X))
    const resultDf = new dfd.DataFrame(scaler.transform(new dfd.DataFrame(X)))
    assert.isTrue(arrayEqual(resultDf.values, expected, 0.1))
  })
})
