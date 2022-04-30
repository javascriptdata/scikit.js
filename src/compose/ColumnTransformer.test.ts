import {
  ColumnTransformer,
  MinMaxScaler,
  SimpleImputer,
  setBackend
} from '../index'
import * as dfd from 'danfojs-node'
import * as tf from '@tensorflow/tfjs-node'
setBackend(tf)

describe('ColumnTransformer', function () {
  it('ColumnTransformer simple test', function () {
    const X = [
      [2, 2], // [1, .5]
      [2, 3], // [1, .75]
      [0, NaN], // [0, 1]
      [2, 0] // [.5, 0]
    ]
    let newDf = new dfd.DataFrame(X)

    const transformer = new ColumnTransformer({
      transformers: [
        ['minmax', new MinMaxScaler(), [0]],
        ['simpleImpute', new SimpleImputer({ strategy: 'median' }), [1]]
      ]
    })

    let result = transformer.fitTransform(newDf)
    const expected = [
      [1, 2],
      [1, 3],
      [0, 2],
      [1, 0]
    ]

    expect(result.arraySync()).toEqual(expected)
  })
})
