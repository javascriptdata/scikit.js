import { OrdinalEncoder } from './ordinalEncoder'
import { arrayTo2DColumn } from '../utils'

describe('OrdinalEncoder', function () {
  it('OrdinalEncoder works on array', function () {
    const data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat']
    const X = arrayTo2DColumn(data)
    const encode = new OrdinalEncoder()
    encode.fit(X)

    const expected = [[0], [1], [2], [0], [1], [2], [2], [1]]
    expect(encode.transform(X).arraySync()).toEqual(expected)
    expect(
      encode.transform(arrayTo2DColumn(['man', 'cat'])).arraySync()
    ).toEqual([[2], [1]])
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
    expect(encode.fitTransform(X as any).arraySync()).toEqual(expected)
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
    expect(encode.fitTransform(X as any).arraySync()).toEqual(expected)
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
    expect(() => encode.transform([['Hello', 1]] as any)).toThrow()
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
    expect(expected.arraySync()).toEqual([[-1, 0]])
  })
})
