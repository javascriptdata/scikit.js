import { assert } from 'chai'
import { LinearSVR } from './linearSVR'
import { describe, it } from 'mocha'
import { tensorEqual } from '../utils'
import { tf } from '../shared/globals'

describe('LinearSVR', function () {
  this.timeout(30000)
  it('Works on arrays (small example)', async function () {
    const lr = new LinearSVR({ epsilon: 0, fitIntercept: false, C: 0.0001 })
    await lr.fit([[1], [2]], [2, 4])
    assert.isTrue(tensorEqual(lr.coef, tf.tensor1d([2]), 0.1))
  })
})
