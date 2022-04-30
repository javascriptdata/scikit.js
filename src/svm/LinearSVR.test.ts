import { LinearSVR, setBackend } from '../index'
import * as tf from '@tensorflow/tfjs-node'
setBackend(tf)
import { tensorEqual } from '../utils'

describe('LinearSVR', function () {
  it('Works on arrays (small example)', async function () {
    const lr = new LinearSVR({ epsilon: 0, fitIntercept: false, C: 0.0001 })
    await lr.fit([[1], [2]], [2, 4])
    expect(tensorEqual(lr.coef, tf.tensor1d([2]), 0.1)).toBe(true)
  }, 30000)
})
