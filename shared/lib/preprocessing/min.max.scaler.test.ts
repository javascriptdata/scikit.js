import { assert } from 'chai'
import { MinMaxScaler } from './min.max.scaler'
import { dfd } from '../../globals'
import { tensor2d } from '@tensorflow/tfjs-core'
import { convertTensorToInputType } from '../utils'

describe('MinMaxscaler', function () {
  it('Standardize values in a DataFrame using a MinMaxScaler', function () {
    const data = [
      [-1, 2],
      [-0.5, 6],
      [0, 10],
      [1, 18]
    ]
    const scaler = new MinMaxScaler()

    const expected = [
      [0, 0],
      [0.25, 0.25],
      [0.5, 0.5],
      [1, 1]
    ]
    const transformedData = [[1.5, 0]]

    scaler.fit(new dfd.DataFrame(data))
    const resultDf = new dfd.DataFrame(
      scaler.transform(new dfd.DataFrame(data))
    )
    assert.deepEqual(resultDf.values, expected)
    assert.deepEqual(scaler.transform([[2, 2]]).arraySync(), transformedData)
  })
  it('fitTransform using a MinMaxScaler', function () {
    const data = [
      [-1, 2],
      [-0.5, 6],
      [0, 10],
      [1, 18]
    ]
    const scaler = new MinMaxScaler()
    const resultDf = new dfd.DataFrame(
      scaler.fitTransform(new dfd.DataFrame(data))
    )

    const expected = [
      [0, 0],
      [0.25, 0.25],
      [0.5, 0.5],
      [1, 1]
    ]
    assert.deepEqual(resultDf.values, expected)
  })
  it('InverseTransform with MinMaxScaler', function () {
    const scaler = new MinMaxScaler()
    const data = tensor2d([1, 2, 3, 4, 5], [5, 1])
    scaler.fit(data)
    const resultTransform = scaler.transform(data)
    const resultInverse = scaler.inverseTransform(
      tensor2d([0, 0.25, 0.5, 0.75, 1], [5, 1])
    )

    assert.deepEqual(
      resultTransform.arraySync().flat(),
      [0, 0.25, 0.5, 0.75, 1]
    )
    assert.deepEqual(data.arraySync(), resultInverse.arraySync())
  })
  it('Index and columns are kept after transformation', function () {
    const data = [
      [-1, 2],
      [-0.5, 6],
      [0, 10],
      [1, 18]
    ]
    const df = new dfd.DataFrame(data, {
      index: [1, 2, 3, 4],
      columns: ['a', 'b']
    })

    const scaler = new MinMaxScaler()
    scaler.fit(df)
    const resultDf = convertTensorToInputType(
      scaler.transform(df),
      df
    ) as dfd.DataFrame

    assert.deepEqual(resultDf.index, [1, 2, 3, 4])
    assert.deepEqual(resultDf.columns, ['a', 'b'])
  })
  it('Handles pathological examples with constant features with MinMaxScaler', function () {
    const data = tensor2d([3, 3, 3, 3, 3], [5, 1])
    const scaler = new MinMaxScaler()
    scaler.fit(data)
    assert.deepEqual(
      scaler
        .transform(tensor2d([0, 1, 10, 10], [4, 1]))
        .arraySync()
        .flat(),
      [-3, -2, 7, 7]
    )
  })
  it('featureRange', function () {
    const data = [
      [-1, 2],
      [-0.5, 6],
      [0, 10],
      [1, 18]
    ]
    let scaler = new MinMaxScaler({ featureRange: [1, 2] })
    const result = scaler.fitTransform(data)
    const expected = [
      [1, 1],
      [1.25, 1.25],
      [1.5, 1.5],
      [2, 2]
    ]
    assert.deepEqual(result.arraySync(), expected)
  })
  it('keeps track of variables', function () {
    let myDf = new dfd.DataFrame({ a: [1, 2, 3, 4], b: [5, 6, 7, 8] })
    let scaler = new MinMaxScaler()
    scaler.fit(myDf)
    assert.deepEqual(scaler.nSamplesSeen, 4)
    assert.deepEqual(scaler.nFeaturesIn, 2)
    assert.deepEqual(scaler.featureNamesIn, ['a', 'b'])
    assert.deepEqual(scaler.dataMin.arraySync(), [1, 5])
    assert.deepEqual(scaler.dataMax.arraySync(), [4, 8])
    assert.deepEqual(scaler.dataRange.arraySync(), [3, 3])
  })
  it('Errors when you pass garbage input into a MinMaxScaler', function () {
    const data = 4
    const scaler = new MinMaxScaler()
    assert.throws(() => scaler.fit(data as any))
  })
  it('Gracefully handles Nan as inputs MinMaxScaler', function () {
    const data = tensor2d([4, 4, 'whoops', 3, 3] as any, [5, 1])
    const scaler = new MinMaxScaler()
    scaler.fit(data)
    assert.deepEqual(scaler.transform(data).arraySync().flat(), [
      1,
      1,
      NaN,
      0,
      0
    ])
  })
})
