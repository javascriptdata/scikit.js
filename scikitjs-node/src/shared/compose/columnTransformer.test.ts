import { assert } from 'chai'
import { ColumnTransformer } from './columnTransformer'
import { MinMaxScaler } from '../preprocessing/minMaxScaler'
import { SimpleImputer } from '../impute/simpleImputer'
import { describe, it } from 'mocha'

describe('ColumnTransformer', function () {
  it('ColumnTransformer simple test', function () {
    const X = [
      [2, 2], // [1, .5]
      [2, 3], // [1, .75]
      [0, NaN], // [0, 1]
      [2, 0] // [.5, 0]
    ]

    const transformer = new ColumnTransformer({
      transformers: [
        ['minmax', new MinMaxScaler(), [0]],
        ['simpleImpute', new SimpleImputer({ strategy: 'median' }), [1]]
      ]
    })

    let result = transformer.fitTransform(X)
    const expected = [
      [1, 2],
      [1, 3],
      [0, 2],
      [1, 0]
    ]
    const save = transformer.toJson()
    console.log(save)
    // assert.deepEqual(result.arraySync(), expected)
  })
})
