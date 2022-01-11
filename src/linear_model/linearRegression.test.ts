import { LinearRegression } from './linearRegression'
import { tensorEqual } from '../utils'
import {
  Tensor1D,
  Tensor2D,
  tensor1d,
  tensor2d,
  randomNormal,
  randomUniform
} from '@tensorflow/tfjs-core'

function roughlyEqual(a: number, b: number, tol = 0.1) {
  return Math.abs(a - b) < tol
}

describe('LinearRegression', function () {
  it('Works on arrays (small example)', async function () {
    const lr = new LinearRegression()
    await lr.fit([[1], [2]], [2, 4])
    expect(tensorEqual(lr.coef, tensor1d([2]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
  }, 30000)

  it('Works on small multi-output example (small example)', async function () {
    const lr = new LinearRegression()
    await lr.fit(
      [[1], [2]],
      [
        [2, 1],
        [4, 2]
      ]
    )

    expect(tensorEqual(lr.coef, tensor2d([[2, 1]]), 0.1)).toBe(true)
  }, 30000)

  it('Works on arrays with no intercept (small example)', async function () {
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit([[1], [2]], [2, 4])
    expect(tensorEqual(lr.coef, tensor1d([2]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
  }, 30000)

  it('Works on arrays with none zero intercept (small example)', async function () {
    const lr = new LinearRegression({ fitIntercept: true })
    await lr.fit([[1], [2]], [3, 5])
    expect(tensorEqual(lr.coef, tensor1d([2]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 1)).toBe(true)
  }, 30000)

  // Medium sized example
  it('Works on arrays with none zero intercept (medium example)', async function () {
    const sizeOfMatrix = 100
    const seed = 42
    let mediumX = randomUniform(
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
      randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    expect(tensorEqual(lr.coef, tensor1d([2.5, 1]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
    expect(lr.score(mediumX, y) > 0).toBe(true)
  }, 30000)

  it('Works on arrays with none zero intercept (medium example)', async function () {
    const sizeOfMatrix = 1000
    const seed = 42
    let mediumX = randomUniform(
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
      randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    expect(tensorEqual(lr.coef, tensor1d([2.5, 1]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
  }, 30000)
})
