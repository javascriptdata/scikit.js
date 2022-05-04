import { MinMaxScaler } from './MinMaxScaler'
import * as dfd from 'danfojs-node'
import { isDataFrameInterface, isSeriesInterface } from '../typesUtils'
import { ScikitVecOrMatrix } from '../types'
import { tf } from '../shared/globals'
import { toObject, fromObject } from '../simpleSerializer'
export function convertTensorToInputType(
  tensor: tf.Tensor,
  inputData: ScikitVecOrMatrix
) {
  if (inputData instanceof tf.Tensor) {
    return tensor
  } else if (isDataFrameInterface(inputData)) {
    return new dfd.DataFrame(tensor, {
      index: inputData.index,
      columns: inputData.columns
    })
  } else if (isSeriesInterface(inputData)) {
    return new dfd.Series(tensor, {
      index: inputData.index
    })
  } else if (Array.isArray(inputData)) {
    return tensor.arraySync()
  } else {
    return tensor
  }
}

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
    expect(resultDf.values).toEqual(expected)
    expect(scaler.transform([[2, 2]]).arraySync()).toEqual(transformedData)
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
    expect(resultDf.values).toEqual(expected)
  })
  it('InverseTransform with MinMaxScaler', function () {
    const scaler = new MinMaxScaler()
    const data = tf.tensor2d([1, 2, 3, 4, 5], [5, 1])
    scaler.fit(data)
    const resultTransform = scaler.transform(data)
    const resultInverse = scaler.inverseTransform(
      tf.tensor2d([0, 0.25, 0.5, 0.75, 1], [5, 1])
    )

    expect(resultTransform.arraySync().flat()).toEqual([0, 0.25, 0.5, 0.75, 1])
    expect(data.arraySync()).toEqual(resultInverse.arraySync())
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

    expect(resultDf.index).toEqual([1, 2, 3, 4])
    expect(resultDf.columns).toEqual(['a', 'b'])
  })
  it('Handles pathological examples with constant features with MinMaxScaler', function () {
    const data = tf.tensor2d([3, 3, 3, 3, 3], [5, 1])
    const scaler = new MinMaxScaler()
    scaler.fit(data)
    expect(
      scaler
        .transform(tf.tensor2d([0, 1, 10, 10], [4, 1]))
        .arraySync()
        .flat()
    ).toEqual([-3, -2, 7, 7])
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
    expect(result.arraySync()).toEqual(expected)
  })
  it('keeps track of variables', function () {
    let myDf = new dfd.DataFrame({ a: [1, 2, 3, 4], b: [5, 6, 7, 8] })
    let scaler = new MinMaxScaler()
    scaler.fit(myDf)
    expect(scaler.nSamplesSeen).toEqual(4)
    expect(scaler.nFeaturesIn).toEqual(2)
    expect(scaler.featureNamesIn).toEqual(['a', 'b'])
    expect(scaler.dataMin.arraySync()).toEqual([1, 5])
    expect(scaler.dataMax.arraySync()).toEqual([4, 8])
    expect(scaler.dataRange.arraySync()).toEqual([3, 3])
  })
  it('Errors when you pass garbage input into a MinMaxScaler', function () {
    const data = 4
    const scaler = new MinMaxScaler()
    expect(() => scaler.fit(data as any)).toThrow()
  })
  it('Gracefully handles Nan as inputs MinMaxScaler', function () {
    const data = tf.tensor2d([4, 4, 'whoops', 3, 3] as any, [5, 1])
    const scaler = new MinMaxScaler()
    scaler.fit(data)
    expect(scaler.transform(data).arraySync().flat()).toEqual([
      1,
      1,
      NaN,
      0,
      0
    ])
  })
  it('Serialize and unserialize MinMaxScaler', async function () {
    const data = tf.tensor2d([4, 4, 'whoops', 3, 3] as any, [5, 1])
    const scaler = new MinMaxScaler()
    scaler.fit(data)
    const serial = (await toObject(scaler)) as string
    const newModel = await fromObject(serial)
    expect(newModel.transform(data).arraySync().flat()).toEqual([
      1,
      1,
      NaN,
      0,
      0
    ])
  })
})
