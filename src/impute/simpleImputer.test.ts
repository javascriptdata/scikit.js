import { tensor2d } from '@tensorflow/tfjs-core'
import { SimpleImputer } from './simpleImputer'

describe('SimpleImputer', function () {
  it('Imputes with "constant" strategy 2D one column. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer({ strategy: 'constant', fillValue: 3 })

    const data = tensor2d([1, 2, NaN, 4, 4], [5, 1])

    const expected = [1, 2, 3, 4, 4]

    const returned = imputer.fitTransform(data)
    expect(returned.arraySync().flat()).toEqual(expected)
    expect(
      imputer
        .transform([[2], [NaN]])
        .arraySync()
        .flat()
    ).toEqual([2, 3])
  })
  it('Imputes with "constant" strategy 2D one column. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer({
      strategy: 'constant',
      fillValue: 4
    })

    const data = tensor2d([
      [1, NaN],
      [4, 4],
      [NaN, 3]
    ])

    const expected = [
      [1, 4],
      [4, 4],
      [4, 3]
    ]

    const returned = imputer.fitTransform(data)
    expect(returned.arraySync()).toEqual(expected)
    expect(
      imputer
        .transform([[NaN, NaN]])
        .arraySync()
        .flat()
    ).toEqual([4, 4])
  })
  it('Imputes with "mean" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer({ strategy: 'mean' })

    const data = [
      [2, 3],
      [NaN, NaN],
      [4, 3]
    ]

    const expected = [
      [2, 3],
      [3, 3],
      [4, 3]
    ]

    const returned = imputer.fitTransform(data)
    expect(returned.arraySync()).toEqual(expected)
    expect(
      imputer
        .transform([
          [2, NaN],
          [NaN, NaN]
        ])
        .arraySync()
    ).toEqual([
      [2, 3],
      [3, 3]
    ])
  })
  it('Imputes with "median" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer({ strategy: 'median' })

    const data = [
      [2, 3],
      [NaN, 3],
      [4, 230],
      [6, NaN]
    ]

    const expected = [
      [2, 3],
      [4, 3],
      [4, 230],
      [6, 3]
    ]

    const returned = imputer.fitTransform(data)
    expect(returned.arraySync()).toEqual(expected)
    expect(imputer.transform([[2, NaN]]).arraySync()).toEqual([[2, 3]])
  })

  it('Imputes with "mostFrequent" strategy. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer({ strategy: 'mostFrequent' })

    const data = [
      [2, 3],
      [NaN, 3],
      [4, 3],
      [4, 2],
      [6, NaN]
    ]

    const expected = [
      [2, 3],
      [4, 3],
      [4, 3],
      [4, 2],
      [6, 3]
    ]

    const returned = imputer.fitTransform(data)
    expect(returned.arraySync()).toEqual(expected)
    expect(imputer.transform([[NaN, NaN]]).arraySync()).toEqual([[4, 3]])
  })
  it('Should serialized Imputer', function () {
    const imputer = new SimpleImputer({ strategy: 'mostFrequent' })

    const data = [
      [2, 3],
      [NaN, 3],
      [4, 3],
      [4, 2],
      [6, NaN]
    ]

    const expected = {
      name: 'simpleimputer',
      missingValues: null,
      strategy: 'mostFrequent',
      statistics: [4, 3]
    }

    const returned = imputer.fitTransform(data)
    expect(JSON.parse(imputer.toJson() as string)).toEqual(expected)
  })
  it('Should load serialized Imputer', function () {
    const imputer = new SimpleImputer({ strategy: 'mostFrequent' })

    const data = [
      [2, 3],
      [NaN, 3],
      [4, 3],
      [4, 2],
      [6, NaN]
    ]

    const expected = [
      [2, 3],
      [4, 3],
      [4, 3],
      [4, 2],
      [6, 3]
    ]

    const returned = imputer.fitTransform(data)
    const newImputer = new SimpleImputer().fromJson(imputer.toJson() as string)
    const newReturned = newImputer.transform(data)
    expect(newReturned.arraySync()).toEqual(expected)
    expect(newImputer.transform([[NaN, NaN]]).arraySync()).toEqual([[4, 3]])
  })
})
