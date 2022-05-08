import { LinearRegression, fromJSON } from '../index'
import { tensorEqual } from '../utils'
import { tf } from '../shared/globals'
function roughlyEqual(a: number, b: number, tol = 0.1) {
  return Math.abs(a - b) < tol
}

describe('LinearRegression', function () {
  it('Works on arrays (small example)', async function () {
    const lr = new LinearRegression()
    await lr.fit([[1], [2]], [2, 4])
    expect(tensorEqual(lr.coef, tf.tensor1d([2]), 0.1)).toBe(true)
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

    expect(tensorEqual(lr.coef, tf.tensor2d([[2, 1]]), 0.1)).toBe(true)
  }, 30000)

  it('Works on arrays with no intercept (small example)', async function () {
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit([[1], [2]], [2, 4])
    expect(tensorEqual(lr.coef, tf.tensor1d([2]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
  }, 30000)

  it('Works on arrays with none zero intercept (small example)', async function () {
    const lr = new LinearRegression({ fitIntercept: true })
    await lr.fit([[1], [2]], [3, 5])
    expect(tensorEqual(lr.coef, tf.tensor1d([2]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 1)).toBe(true)
  }, 30000)

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
    ) as tf.Tensor2D
    let [firstCol, secondCol] = mediumX.split([1, 1], 1)
    let y = firstCol
      .mul(2.5)
      .add(secondCol)
      .reshape([sizeOfMatrix]) as tf.Tensor1D
    const yPlusJitter = y.add(
      tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as tf.Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    expect(tensorEqual(lr.coef, tf.tensor1d([2.5, 1]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
    expect(lr.score(mediumX, y) > 0).toBe(true)
  }, 30000)

  it('Works on arrays with none zero intercept (medium example)', async function () {
    const sizeOfMatrix = 1000
    const seed = 42
    let mediumX = tf.randomUniform(
      [sizeOfMatrix, 2],
      -10,
      10,
      'float32',
      seed
    ) as tf.Tensor2D
    let [firstCol, secondCol] = mediumX.split([1, 1], 1)
    let y = firstCol
      .mul(2.5)
      .add(secondCol)
      .reshape([sizeOfMatrix]) as tf.Tensor1D
    const yPlusJitter = y.add(
      tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as tf.Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    expect(tensorEqual(lr.coef, tf.tensor1d([2.5, 1]), 0.1)).toBe(true)
    expect(roughlyEqual(lr.intercept as number, 0)).toBe(true)
  }, 30000)
  it('Cog python test', async function () {
    /*
      [[[cog
      import cog
      import numpy as np

      def print_numpy(X):
        return repr(X).replace('array(', '').replace(')', '')

      np.set_printoptions(threshold=np.inf)
      from sklearn.datasets import make_regression
      from sklearn.linear_model import LinearRegression
      X, y = make_regression(5, 5, n_informative=5)

      lf = LinearRegression()
      lf.fit(X, y)
      cog.outl("let X = " + print_numpy(X))
      cog.outl("let y = " + print_numpy(y))
      cog.outl("let score = " + print_numpy(lf.score(X, y)))
      ]]]*/
    let X = [
      [-0.00244914, 0.06518702, -1.36608168, -0.5427704, 1.48407056],
      [1.30791547, -0.70896184, 1.51020065, 0.06197986, 2.28595886],
      [-0.17730942, -1.09547718, 0.29676505, 1.19286185, -1.88178388],
      [1.20533452, -0.53933888, 0.20527909, -0.79376298, -0.5496481],
      [-0.03124791, -0.3598143, 1.74913615, -0.89918996, -0.21714292]
    ]
    let y = [-70.62017777, 130.65722027, 18.93268378, 13.60904025, 48.96704418]
    let score = 1.0
    /*[[[end]]]*/

    const lr = new LinearRegression()
    await lr.fit(X, y)
    expect(lr.score(X, y)).toBeCloseTo(score)
  }, 30000)
  it('Should save and load Model using arrays with none zero intercept (medium example)', async function () {
    const sizeOfMatrix = 1000
    const seed = 42
    let mediumX = tf.randomUniform(
      [sizeOfMatrix, 2],
      -10,
      10,
      'float32',
      seed
    ) as tf.Tensor2D
    let [firstCol, secondCol] = mediumX.split([1, 1], 1)
    let y = firstCol
      .mul(2.5)
      .add(secondCol)
      .reshape([sizeOfMatrix]) as tf.Tensor1D
    const yPlusJitter = y.add(
      tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
    ) as tf.Tensor1D
    const lr = new LinearRegression({ fitIntercept: false })
    await lr.fit(mediumX, yPlusJitter)

    const serialized = await lr.toJSON()
    const newModel = await fromJSON(serialized)

    expect(tensorEqual(newModel.coef, tf.tensor1d([2.5, 1]), 0.1)).toBe(true)
    expect(roughlyEqual(newModel.intercept as number, 0)).toBe(true)
  }, 30000)
})
