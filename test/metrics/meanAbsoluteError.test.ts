import { assert } from 'chai'
import { meanAbsoluteError } from '../../dist/metrics/meanAbsoluteError'

describe('Mean Absolute Error', function () {
  it('Small input sanity check', function () {
    const newTensor = meanAbsoluteError([0, 1], [0, 0])
    assert.isTrue(newTensor.shape.length === 0)
    assert.isTrue(newTensor.dataSync()[0] === 0.5)
    newTensor.dispose()
  })
})
