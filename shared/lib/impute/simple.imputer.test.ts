import { tensor2d } from '@tensorflow/tfjs-core'
import { assert } from 'chai'
import SimpleImputer from './simple.imputer'

describe('SimpleImputer', function () {
  it('Imputes with "constant" strategy 2D one column. In this strategy, we give the fill value', function () {
    const imputer = new SimpleImputer({ strategy: 'constant', fillValue: 3 })

    const data = tensor2d([1, 2, NaN, 4, 4], [5, 1])

    const expected = [1, 2, 3, 4, 4]

    const returned = imputer.fitTransform(data)
    assert.deepEqual(returned.arraySync().flat(), expected)
    assert.deepEqual(
      imputer
        .transform([[2], [NaN]])
        .arraySync()
        .flat(),
      [2, 3]
    )
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
    assert.deepEqual(returned.arraySync(), expected)
    assert.deepEqual(
      imputer
        .transform([[NaN, NaN]])
        .arraySync()
        .flat(),
      [4, 4]
    )
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
    assert.deepEqual(returned.arraySync(), expected)
    assert.deepEqual(
      imputer
        .transform([
          [2, NaN],
          [NaN, NaN]
        ])
        .arraySync(),
      [
        [2, 3],
        [3, 3]
      ]
    )
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
    assert.deepEqual(returned.arraySync(), expected)
    assert.deepEqual(imputer.transform([[2, NaN]]).arraySync(), [[2, 3]])
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
    assert.deepEqual(returned.arraySync(), expected)
    assert.deepEqual(imputer.transform([[NaN, NaN]]).arraySync(), [[4, 3]])
  })
})
