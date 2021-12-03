import { assert } from 'chai'
import { OrdinalEncoder } from './ordinal.encoder'
import { arrayTo2DColumn } from '../utils'
import { describe, it } from 'mocha'

describe('OrdinalEncoder', function () {
  it('OrdinalEncoder works on array', function () {
    const data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat']
    const X = arrayTo2DColumn(data)
    const encode = new OrdinalEncoder()
    encode.fit(X)

    const expected = [[0], [1], [2], [0], [1], [2], [2], [1]]
    assert.deepEqual(encode.transform(X).arraySync(), expected)
    assert.deepEqual(
      encode.transform(arrayTo2DColumn(['man', 'cat'])).arraySync(),
      [[2], [1]]
    )
  })
  it('OrdinalEncoder works on 2DArray', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OrdinalEncoder()

    const expected = [
      [0, 0],
      [1, 1],
      [0, 2]
    ]
    assert.deepEqual(encode.fitTransform(X as any).arraySync(), expected)
  })
  it('OrdinalEncoder can be passed categories', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OrdinalEncoder({
      categories: [
        ['Male', 'Female'],
        [4, 2, 1]
      ]
    })

    const expected = [
      [0, 2],
      [1, 1],
      [0, 0]
    ]
    assert.deepEqual(encode.fitTransform(X as any).arraySync(), expected)
  })
  it('OrdinalEncoder errors on values not seen in training', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OrdinalEncoder()
    encode.fit(X as any)
    // Should throw an error on unknown input
    assert.throw(() => encode.transform([['Hello', 1]] as any))
  })
  it('OrdinalEncoder does not error on unknown values if you pass in defaults', function () {
    const X = [
      ['Male', 1],
      ['Female', 2],
      ['Male', 4]
    ]
    const encode = new OrdinalEncoder({
      handleUnknown: 'useEncodedValue',
      unknownValue: -1
    })
    encode.fit(X as any)
    // Should throw an error on unknown input
    const expected = encode.transform([['Hello', 1]] as any)
    assert.deepEqual(expected.arraySync(), [[-1, 0]])
  })
})
