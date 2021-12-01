import { assert } from 'chai'
import { Pipeline } from './pipeline'
import { tensor1d } from '@tensorflow/tfjs-core'
import { tensorEqual } from '../utils'
import { LinearRegression } from '../estimators/linear.regression'
import { SimpleImputer } from '../impute/simple.imputer'
import { MinMaxScaler } from '../preprocessing/min.max.scaler'

describe('Pipeline', function () {
  it('Use a Pipeline (min-max scaler, and linear regression)', async function () {
    const X = [
      [2, 2], // [1, .5]
      [2, 3], // [1, .75]
      [0, 4], // [0, 1]
      [1, 0] // [.5, 0]
    ]
    const y = [5, 6, 4, 1.5]
    const pipeline = new Pipeline({
      steps: [
        ['minmax', new MinMaxScaler()],
        ['lr', new LinearRegression({ fitIntercept: false })]
      ]
    })

    await pipeline.fit(X, y)

    assert.deepEqual(pipeline.steps[0][1].min.arraySync(), [0, 0])
    assert.deepEqual(
      tensorEqual(pipeline.steps[1][1].coef, tensor1d([3, 4]), 0.3),
      true
    )
  })
  it('Use a Pipeline (simple-imputer, min-max, linear regression)', async function () {
    const X = [
      [2, 2], // [1, .5]
      [2, NaN], // [1, 0]
      [NaN, 4], // [0, 1]
      [1, 0] // [.5, 0]
    ]
    const y = [5, 3, 4, 1.5]
    const pipeline = new Pipeline({
      steps: [
        [
          'simpleImputer',
          new SimpleImputer({ strategy: 'constant', fillValue: 0 })
        ],
        ['minmax', new MinMaxScaler()],
        ['lr', new LinearRegression({ fitIntercept: false })]
      ]
    })

    await pipeline.fit(X, y)

    assert.deepEqual(pipeline.steps[1][1].min.arraySync(), [0, 0])
    assert.deepEqual(
      tensorEqual(pipeline.steps[2][1].coef, tensor1d([3, 4]), 0.3),
      true
    )
  })
})
