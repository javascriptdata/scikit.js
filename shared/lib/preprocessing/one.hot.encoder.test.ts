import { tensor2d } from '@tensorflow/tfjs-core'
import { assert } from 'chai'
import { OneHotEncoder } from './one.hot.encoder'
import { arrayTo2DColumn } from '../utils'
import { describe, it } from 'mocha'

describe('OneHotEncoder', function () {
  it('OneHotEncoder works on array', function () {
    const data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat']
    const X = arrayTo2DColumn(data)
    const encode = new OneHotEncoder()
    encode.fit(X)

    const expected = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 1],
      [0, 1, 0]
    ]
    assert.deepEqual(encode.transform(X).arraySync(), expected)
    assert.deepEqual(
      encode.transform(arrayTo2DColumn(['man', 'cat'])).arraySync(),
      [
        [0, 0, 1],
        [0, 1, 0]
      ]
    )
    assert.deepEqual(
      encode.inverseTransform(
        tensor2d([
          [0, 0, 1],
          [0, 1, 0]
        ])
      ),
      ['man', 'cat']
    )
  })
  it('OneHotEncoder works on 2DArray', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OneHotEncoder()

    const expected = [
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 0],
      [1, 0, 0, 0, 1]
    ]
    assert.deepEqual(encode.fitTransform(X as any).arraySync(), expected)
  })
})
