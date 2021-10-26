import { assert } from 'chai'
import { RobustScaler } from '../../../dist'
import { DataFrame } from 'danfojs-node'
import { tensor2d } from '@tensorflow/tfjs-core'
import { convertTensorToInputType } from '../../utils'

describe('RobustScaler', function () {
  it('Standardize values in a DataFrame using a RobustScaler', function () {
    const X = [
      [1, -2, 2],
      [-2, 1, 3],
      [4, 1, -2]
    ]

    const scaler = new RobustScaler()

    const expected = [
      [0, -2, 0],
      [-1, 0, 0.4],
      [1, 0, -1.6]
    ]

    scaler.fit(new DataFrame(X))
    const resultDf = new DataFrame(scaler.transform(new DataFrame(X)))
    console.log(resultDf.values)
    assert.deepEqual(resultDf.values, expected)
  })
  // it('fitTransform using a RobustScaler', function () {
  //   const data = [
  //     [-1, 2],
  //     [-0.5, 6],
  //     [0, 10],
  //     [1, 18]
  //   ]
  //   const scaler = new RobustScaler()
  //   const resultDf = new DataFrame(scaler.fitTransform(new DataFrame(data)))

  //   const expected = [
  //     [0, 0],
  //     [0.25, 0.25],
  //     [0.5, 0.5],
  //     [1, 1]
  //   ]
  //   assert.deepEqual(resultDf.values, expected)
  // })
  // it('InverseTransform with RobustScaler', function () {
  //   const scaler = new RobustScaler()
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

  //   const scaler = new RobustScaler()
  //   scaler.fit(df)
  //   const resultDf = convertTensorToInputType(
  //     scaler.transform(df),
  //     df
  //   ) as DataFrame

  //   assert.deepEqual(resultDf.index, [1, 2, 3, 4])
  //   assert.deepEqual(resultDf.columns, ['a', 'b'])
  // })
  // it('Handles pathological examples with constant features with RobustScaler', function () {
  //   const data = tensor2d([3, 3, 3, 3, 3], [5, 1])
  //   const scaler = new RobustScaler()
  //   scaler.fit(data)
  //   assert.deepEqual(
  //     scaler
  //       .transform(tensor2d([0, 1, 10, 10], [4, 1]))
  //       .arraySync()
  //       .flat(),
  //     [-3, -2, 7, 7]
  //   )
  // })
  // it('Errors when you pass garbage input into a RobustScaler', function () {
  //   const data = 4
  //   const scaler = new RobustScaler()
  //   assert.throws(() => scaler.fit(data as any))
  // })
  // it('Gracefully handles Nan as inputs RobustScaler', function () {
  //   const data = tensor2d([4, 4, 'whoops', 3, 3] as any, [5, 1])
  //   const scaler = new RobustScaler()
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
