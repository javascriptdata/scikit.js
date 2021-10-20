import { assert } from 'chai'
import { SimpleImputer } from '../../dist'
import { Series, DataFrame } from 'danfojs-node'

describe('SimpleImputer', function () {
  it('Imputes with "constant" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer('constant', 3)

    const data = [1, 2, NaN, 4, 4]

    const expected = [1, 2, 3, 4, 4]

    const returned = imputer.fitTransform(data)
    assert.deepEqual(returned, expected)
    assert.deepEqual(imputer.transform([2, NaN]), [2, 3])
  })
  it('Imputes with "mean" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer('mean')

    const data = [2, NaN, 4]

    const expected = [2, 3, 4]

    const returned = imputer.fitTransform(data)
    assert.deepEqual(returned, expected)
    assert.deepEqual(imputer.transform([2, NaN]), [2, 3])
  })
  it('Imputes with "median" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer('mean')

    const data = [2, NaN, 4, 6]

    const expected = [2, 4, 4, 6]

    const returned = imputer.fitTransform(data)
    assert.deepEqual(returned, expected)
    assert.deepEqual(imputer.transform([2, NaN]), [2, 4])
  })

  it('Imputes with "mostFrequent" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer('mean')

    const data = [2, NaN, 4, 4, 6]

    const expected = [2, 4, 4, 4, 6]

    const returned = imputer.fitTransform(data)
    assert.deepEqual(returned, expected)
    assert.deepEqual(imputer.transform([2, NaN]), [2, 4])
  })
})
