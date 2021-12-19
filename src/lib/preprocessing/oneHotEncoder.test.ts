import { tensor2d } from '@tensorflow/tfjs-core'
import { assert } from 'chai'
import { OneHotEncoder } from './oneHotEncoder'
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
  it('OneHotEncoder can be passed categories', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OneHotEncoder({
      categories: [
        ['Male', 'Female'],
        [4, 2, 1]
      ]
    })

    const expected = [
      [1, 0, 0, 0, 1],
      [0, 1, 0, 1, 0],
      [1, 0, 1, 0, 0]
    ]
    assert.deepEqual(encode.fitTransform(X as any).arraySync(), expected)
  })
  it('OneHotEncoder errors on values not seen in training', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OneHotEncoder()
    encode.fit(X as any)
    // Should throw an error on unknown input
    assert.throw(() => encode.transform([['Hello', 1]] as any))
  })
  it('OneHotEncoder does not error on unknown values if you ignore errors', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OneHotEncoder({
      handleUnknown: 'ignore'
    })
    encode.fit(X as any)
    // Should throw an error on unknown input
    const expected = encode.transform([['Hello', 1]] as any)
    assert.deepEqual(expected.arraySync(), [[0, 0, 1, 0, 0]])
  })
  it('OneHotEncoder drop option first', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OneHotEncoder({
      drop: 'first'
    })
    encode.fit(X as any)
    // Should throw an error on unknown input
    const expected = encode.transform([
      ['Male', 1],
      ['Female', 4]
    ] as any)
    assert.deepEqual(expected.arraySync(), [
      [0, 0, 0],
      [1, 0, 1]
    ])
  })
})
