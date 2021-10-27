import { assert } from 'chai'
import { Normalizer } from '../../../dist'
import { DataFrame } from 'danfojs-node'
import { tensor2d } from '@tensorflow/tfjs-core'
import { arrayEqual } from '../../utils'

describe('Normalizer', function () {
  it('Standardize values in a DataFrame using a Normalizer (l1 case)', function () {
    const data = [
      [-1, 1],
      [-6, 6],
      [0, 10],
      [10, 20]
    ]
    const scaler = new Normalizer('l1')

    const expected = [
      [-0.5, 0.5],
      [-0.5, 0.5],
      [0, 1],
      [0.33, 0.66]
    ]
    const transformedData = [[0.5, 0.5]]

    scaler.fit(new DataFrame(data))
    const resultDf = new DataFrame(scaler.transform(new DataFrame(data)))
    assert.isTrue(arrayEqual(resultDf.values, expected, 0.1))
    assert.deepEqual(scaler.transform([[2, 2]]).arraySync(), transformedData)
  })
  it('fitTransform using a Normalizer (l2 case)', function () {
    const data = [
      [-1, 2],
      [-3, 4],
      [0, 10]
    ]
    const scaler = new Normalizer()
    const resultDf = scaler.fitTransform(data)

    const expected = [
      [-1 / Math.sqrt(5), 2 / Math.sqrt(5)],
      [-0.6, 0.8],
      [0, 1]
    ]
    assert.isTrue(arrayEqual(resultDf.arraySync(), expected, 0.1))
  })
  it('fitTransform using a Normalizer (max case)', function () {
    const data = [
      [-1, 2],
      [-3, 4],
      [0, 10]
    ]
    const scaler = new Normalizer('max')
    const resultDf = scaler.fitTransform(data)

    const expected = [
      [-0.5, 1],
      [-0.75, 1],
      [0, 1]
    ]
    assert.isTrue(arrayEqual(resultDf.arraySync(), expected, 0.1))
  })
  // it('InverseTransform with Normalizer', function () {
  //   const scaler = new Normalizer()
  //   const data = tensor2d([1, 2, 3, 4, 5], [5, 1])
  //   scaler.fit(data)
  //   const resultTransform = scaler.transform(data)
  //   const resultInverse = scaler.inverseTransform(
  //     tensor2d([0, 0.25, 0.5, 0.75, 1], [5, 1])
  //   )

  //   assert.deepEqual(
  //     resultTransform.arraySync().flat(),
  //     [0, 0.25, 0.5, 0.75, 1]
  //   )
  //   assert.deepEqual(data.arraySync(), resultInverse.arraySync())
  // })
  // it('Index and columns are kept after transformation', function () {
  //   const data = [
  //     [-1, 2],
  //     [-0.5, 6],
  //     [0, 10],
  //     [1, 18]
  //   ]
  //   const df = new DataFrame(data, {
  //     index: [1, 2, 3, 4],
  //     columns: ['a', 'b']
  //   })

  //   const scaler = new Normalizer()
  //   scaler.fit(df)
  //   const resultDf = convertTensorToInputType(
  //     scaler.transform(df),
  //     df
  //   ) as DataFrame

  //   assert.deepEqual(resultDf.index, [1, 2, 3, 4])
  //   assert.deepEqual(resultDf.columns, ['a', 'b'])
  // })
  // it('Handles pathological examples with constant features with Normalizer', function () {
  //   const data = tensor2d([3, 3, 3, 3, 3], [5, 1])
  //   const scaler = new Normalizer()
  //   scaler.fit(data)
  //   assert.deepEqual(
  //     scaler
  //       .transform(tensor2d([0, 1, 10, 10], [4, 1]))
  //       .arraySync()
  //       .flat(),
  //     [-3, -2, 7, 7]
  //   )
  // })
  // it('Errors when you pass garbage input into a Normalizer', function () {
  //   const data = 4
  //   const scaler = new Normalizer()
  //   assert.throws(() => scaler.fit(data as any))
  // })
  // it('Gracefully handles Nan as inputs Normalizer', function () {
  //   const data = tensor2d([4, 4, 'whoops', 3, 3] as any, [5, 1])
  //   const scaler = new Normalizer()
  //   scaler.fit(data)
  //   assert.deepEqual(scaler.transform(data).arraySync().flat(), [
  //     1,
  //     1,
  //     NaN,
  //     0,
  //     0
  //   ])
  // })
})
