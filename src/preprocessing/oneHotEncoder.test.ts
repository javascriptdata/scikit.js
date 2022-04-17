import { tf } from '../shared/globals'
import { OneHotEncoder } from './oneHotEncoder'
import { arrayTo2DColumn } from '../utils'

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
    expect(encode.transform(X).arraySync()).toEqual(expected)
    expect(
      encode.transform(arrayTo2DColumn(['man', 'cat'])).arraySync()
    ).toEqual([
      [0, 0, 1],
      [0, 1, 0]
    ])
    expect(
      encode.inverseTransform(
        tf.tensor2d([
          [0, 0, 1],
          [0, 1, 0]
        ])
      )
    ).toEqual(['man', 'cat'])
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
    expect(encode.fitTransform(X as any).arraySync()).toEqual(expected)
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
    expect(encode.fitTransform(X as any).arraySync()).toEqual(expected)
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
    expect(() => encode.transform([['Hello', 1]] as any)).toThrow()
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
    expect(expected.arraySync()).toEqual([[0, 0, 1, 0, 0]])
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
    expect(expected.arraySync()).toEqual([
      [0, 0, 0],
      [1, 0, 1]
    ])
  })
})
