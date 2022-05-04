import { Pipeline, makePipeline } from './Pipeline'
import { tf } from '../shared/globals'
import { tensorEqual } from '../utils'
import { LinearRegression } from '../linear_model/LinearRegression'
import { SimpleImputer } from '../impute/SimpleImputer'
import { MinMaxScaler } from '../preprocessing/MinMaxScaler'
import { toObject, fromObject } from '../simpleSerializer'

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

    expect(pipeline.steps[0][1].min.arraySync()).toEqual([0, 0])
    expect(
      tensorEqual(pipeline.steps[1][1].coef, tf.tensor1d([3, 4]), 0.3)
    ).toEqual(true)
  }, 4000)
  it('Use a Pipeline (simple-imputer, min-max, linear regression)', async function () {
    const X = [
      [2, 2], // [1, .5]
      [2, NaN], // [1, 0]
      [NaN, 4], // [0, 1]
      [1, 0] // [.5, 0]
    ]
    const y = [5, 3, 4, 1.5]
    const pipeline = makePipeline(
      new SimpleImputer({ strategy: 'constant', fillValue: 0 }),
      new MinMaxScaler(),
      new LinearRegression({ fitIntercept: false })
    )

    await pipeline.fit(X, y)

    expect(pipeline.steps[1][1].min.arraySync()).toEqual([0, 0])
    expect(
      tensorEqual(pipeline.steps[2][1].coef, tf.tensor1d([3, 4]), 0.3)
    ).toEqual(true)
  }, 4000)
  it('Use makePipeline (simple-imputer, min-max, linear regression)', async function () {
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

    expect(pipeline.steps[1][1].min.arraySync()).toEqual([0, 0])
    expect(
      tensorEqual(pipeline.steps[2][1].coef, tf.tensor1d([3, 4]), 0.3)
    ).toEqual(true)
  }, 4000)
  it('Save and Load Pipeline', async function () {
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

    const saveModel = await toObject(pipeline)
    console.log(saveModel)
    const newPipeLine = await fromObject(saveModel)

    expect(newPipeLine.steps[1][1].min.arraySync()).toEqual([0, 0])
    expect(
      tensorEqual(newPipeLine.steps[2][1].coef, tf.tensor1d([3, 4]), 0.3)
    ).toEqual(true)
  }, 4000)
  it('Make sure the pipeline throws on bad input', async function () {
    expect(
      () =>
        new Pipeline({
          steps: [null] as any
        })
    ).toThrow()
    expect(
      () =>
        new Pipeline({
          steps: null as any
        })
    ).toThrow()
    expect(
      () =>
        new Pipeline({
          steps: [4, 5] as any
        })
    ).toThrow()
    expect(
      () =>
        new Pipeline({
          steps: [new MinMaxScaler()] as any
        })
    ).toThrow()
    expect(
      () =>
        new Pipeline({
          steps: [
            ['minmaxscaler', new MinMaxScaler()],
            [new MinMaxScaler()]
          ] as any
        })
    ).toThrow()
  })
})
