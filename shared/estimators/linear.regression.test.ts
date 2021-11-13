import '@tensorflow/tfjs-backend-webgl'
import * as tf from '@tensorflow/tfjs-core'
import { assert } from 'chai'
import LinearRegression from './linear.regression'
import 'mocha'
import { tensorEqual } from '../utils'
import { Tensor1D, Tensor2D } from '@tensorflow/tfjs-core'

function roughlyEqual(a: number, b: number, tol = 0.1) {
  return Math.abs(a - b) < tol
}

describe('LinearRegression', function () {
  this.timeout(30000)
  it('Works on arrays (small example)', async function () {
    const lr = new LinearRegression()
    await lr.fit([[1], [2]], [2, 4])
    assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2]), 0.1))
    assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  })

  it('Works on small multi-output example (small example)', async function () {
    const lr = new LinearRegression()
    await lr.fit(
      [[1], [2]],
      [
        [2, 1],
        [4, 2]
      ]
    )

    assert.isTrue(tensorEqual(lr.coef_, tf.tensor2d([[2, 1]]), 0.1))
  })

  it('Works on arrays with no intercept (small example)', async function () {
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit([[1], [2]], [2, 4])
    assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2]), 0.1))
    assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  })

  it('Works on arrays with none zero intercept (small example)', async function () {
    const lr = new LinearRegression({ fitIntercept: true })
    await lr.fit([[1], [2]], [3, 5])
    assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2]), 0.1))
    assert.isTrue(roughlyEqual(lr.intercept_ as number, 1))
  })

  // Medium sized example
  it('Works on arrays with none zero intercept (medium example)', async function () {
    const sizeOfMatrix = 100
    const seed = 42
    let mediumX = tf.randomUniform(
      [sizeOfMatrix, 2],
      -10,
      10,
      'float32',
      seed
    ) as Tensor2D
    let [firstCol, secondCol] = mediumX.split([1, 1], 1)
    let y = firstCol
      .mul(2.5)
      .add(secondCol)
      .reshape([sizeOfMatrix]) as Tensor1D
    const yPlusJitter = y.add(
      tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2.5, 1]), 0.1))
    assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  })

  it('Works on arrays with none zero intercept (medium example)', async function () {
    const sizeOfMatrix = 1000
    const seed = 42
    let mediumX = tf.randomUniform(
      [sizeOfMatrix, 2],
      -10,
      10,
      'float32',
      seed
    ) as Tensor2D
    let [firstCol, secondCol] = mediumX.split([1, 1], 1)
    let y = firstCol
      .mul(2.5)
      .add(secondCol)
      .reshape([sizeOfMatrix]) as Tensor1D
    const yPlusJitter = y.add(
      tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2.5, 1]), 0.1))
    assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  })
})
