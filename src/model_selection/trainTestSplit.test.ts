import { assert } from 'chai'
import {
  trainTestSplit,
  validateShuffleSplit,
  getIndices
} from './trainTestSplit'
import { describe, it } from 'mocha'
import { dfd, tf } from '../shared/globals'

describe('Testing trainTestSplit', function () {
  it('Testing train/test validation logic', () => {
    assert.throws(() => validateShuffleSplit(10, 11))
    assert.throws(() => validateShuffleSplit(10, undefined, 100))
    assert.throws(() => validateShuffleSplit(10, 3.5))
    assert.throws(() => validateShuffleSplit(10, 0.3, 0.8))
    assert.throws(() => validateShuffleSplit(10, null as any))
    assert.throws(() => validateShuffleSplit(10, 0))
    assert.throws(() => validateShuffleSplit(10, 5, 6))
    assert.throws(() => validateShuffleSplit(10, {} as any))
    assert.throws(() => validateShuffleSplit(null as any, 11))
  })
  it('Testing train/test acceptance logic', () => {
    let val1 = validateShuffleSplit(10, 3)
    assert.deepEqual(val1, [7, 3])

    let val2 = validateShuffleSplit(10, undefined, 3)
    assert.deepEqual(val2, [3, 7])

    let val3 = validateShuffleSplit(10, 0.1)
    assert.deepEqual(val3, [9, 1])

    let val4 = validateShuffleSplit(10, 0.26)
    assert.deepEqual(val4, [7, 3])

    let val5 = validateShuffleSplit(10, undefined, 0.1)
    assert.deepEqual(val5, [1, 9])

    let val6 = validateShuffleSplit(10, undefined, 0.26)
    assert.deepEqual(val6, [3, 7])

    let val7 = validateShuffleSplit(12, 0.1)
    assert.deepEqual(val7, [10, 2])
  })

  it('Testing getIndices logic', () => {
    let val1 = getIndices([1, 2, 3, 4], [1, 2, 3, 0])
    assert.deepEqual(val1, [2, 3, 4, 1])

    let val2 = getIndices(
      [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
      ],
      [1, 2, 3, 0]
    )
    assert.deepEqual(val2, [
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

    let val3 = getIndices(X, [1, 2, 0]) as dfd.DataFrame
    assert.deepEqual(val3.values, [
      [1, 2],
      [2, 3],
      [0, 1]
    ])

    let X1 = new dfd.Series([0, 1, 2])

    let val4 = getIndices(X1, [1, 2, 0]) as dfd.DataFrame
    assert.deepEqual(val4.values, [1, 2, 0])

    let X2D = tf.tensor2d([
      [1, 2],
      [2, 3],
      [3, 4]
    ])

    let val5 = getIndices(X2D, [1, 2, 0]) as dfd.DataFrame
    assert.deepEqual(val5.arraySync(), [
      [2, 3],
      [3, 4],
      [1, 2]
    ])

    let X1D = tf.tensor1d([1, 2, 3, 4, 5])

    let val6 = getIndices(X1D, [1, 2, 0, 3, 4]) as dfd.DataFrame
    assert.deepEqual(val6.arraySync(), [2, 3, 1, 4, 5])
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
    assert.equal((XTrain as any[]).length, 2)
    assert.equal((XTest as any[]).length, 1)
    assert.equal((yTrain as any[]).length, 2)
    assert.equal((yTest as any[]).length, 1)
    assert.deepEqual(XTrain, [
      [2, 3],
      [3, 4]
    ])
    assert.deepEqual(XTest, [[1, 2]])
    assert.deepEqual(yTrain, [20, 30])
    assert.deepEqual(yTest, [10])
  })
})
