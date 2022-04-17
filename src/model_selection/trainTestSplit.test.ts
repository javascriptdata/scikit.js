import {
  trainTestSplit,
  validateShuffleSplit,
  getIndices
} from './trainTestSplit'

import * as dfd from 'danfojs'
import { tf } from '../shared/globals'
import { DataFrameInterface } from '../types'

describe('Testing trainTestSplit', function () {
  it('Testing train/test validation logic', () => {
    expect(() => validateShuffleSplit(10, 11)).toThrow()
    expect(() => validateShuffleSplit(10, undefined, 100)).toThrow()
    expect(() => validateShuffleSplit(10, 3.5)).toThrow()
    expect(() => validateShuffleSplit(10, 0.3, 0.8)).toThrow()
    expect(() => validateShuffleSplit(10, null as any)).toThrow()
    expect(() => validateShuffleSplit(10, 0)).toThrow()
    expect(() => validateShuffleSplit(10, 5, 6)).toThrow()
    expect(() => validateShuffleSplit(10, {} as any)).toThrow()
    expect(() => validateShuffleSplit(null as any, 11)).toThrow()
  })
  it('Testing train/test acceptance logic', () => {
    let val1 = validateShuffleSplit(10, 3)
    expect(val1).toEqual([7, 3])

    let val2 = validateShuffleSplit(10, undefined, 3)
    expect(val2).toEqual([3, 7])

    let val3 = validateShuffleSplit(10, 0.1)
    expect(val3).toEqual([9, 1])

    let val4 = validateShuffleSplit(10, 0.26)
    expect(val4).toEqual([7, 3])

    let val5 = validateShuffleSplit(10, undefined, 0.1)
    expect(val5).toEqual([1, 9])

    let val6 = validateShuffleSplit(10, undefined, 0.26)
    expect(val6).toEqual([3, 7])

    let val7 = validateShuffleSplit(12, 0.1)
    expect(val7).toEqual([10, 2])
  })

  it('Testing getIndices logic', () => {
    let val1 = getIndices([1, 2, 3, 4], [1, 2, 3, 0])
    expect(val1).toEqual([2, 3, 4, 1])

    let val2 = getIndices(
      [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
      ],
      [1, 2, 3, 0]
    )
    expect(val2).toEqual([
      [2, 3],
      [3, 4],
      [4, 5],
      [1, 2]
    ])

    let X = new dfd.DataFrame([
      [0, 1],
      [1, 2],
      [2, 3]
    ])

    let val3 = getIndices(X as any, [1, 2, 0]) as DataFrameInterface
    expect(val3.values).toEqual([
      [1, 2],
      [2, 3],
      [0, 1]
    ])

    let X1 = new dfd.Series([0, 1, 2])

    let val4 = getIndices(X1 as any, [1, 2, 0]) as DataFrameInterface
    expect(val4.values).toEqual([1, 2, 0])

    let X2D = tf.tensor2d([
      [1, 2],
      [2, 3],
      [3, 4]
    ])

    let val5 = getIndices(X2D as any, [1, 2, 0]) as DataFrameInterface
    expect(val5.arraySync()).toEqual([
      [2, 3],
      [3, 4],
      [1, 2]
    ])

    let X1D = tf.tensor1d([1, 2, 3, 4, 5])

    let val6 = getIndices(X1D as any, [1, 2, 0, 3, 4]) as DataFrameInterface
    expect(val6.arraySync()).toEqual([2, 3, 1, 4, 5])
  })
  it('trainTestSplit indices', () => {
    let X = [
      [1, 2],
      [2, 3],
      [3, 4]
    ]
    let y = [10, 20, 30]

    let [XTrain, XTest, yTrain, yTest] = trainTestSplit(
      X,
      y,
      0.3,
      undefined,
      0
    )
    expect((XTrain as any[]).length).toEqual(2)
    expect((XTest as any[]).length).toEqual(1)
    expect((yTrain as any[]).length).toEqual(2)
    expect((yTest as any[]).length).toEqual(1)
    expect(XTrain).toEqual([
      [2, 3],
      [3, 4]
    ])
    expect(XTest).toEqual([[1, 2]])
    expect(yTrain).toEqual([20, 30])
    expect(yTest).toEqual([10])
  })
})
