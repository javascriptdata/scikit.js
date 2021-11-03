import '@tensorflow/tfjs-backend-webgl'
import * as tf from '@tensorflow/tfjs-core'
import { assert } from 'chai'
import { LogisticRegression } from '../../dist'
import 'mocha'
import { arrayTo2DColumn, tensorEqual } from '../utils'

function roughlyEqual(a: number, b: number, tol = 0.1) {
  return Math.abs(a - b) < tol
}

describe('LogisticRegression', function () {
  this.timeout(30000)
  it('Works on arrays (small example)', async function () {
    const lr = new LogisticRegression()

    await lr.fit([[1], [2]], [0, 1])
    assert.deepEqual(lr.predict([[1], [2]]).arraySync(), [[0], [1]])
  })
  it('Test of the function used with 2 classes', async function () {
    let X = [
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4]
    ]
    let y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    let Xtest = [
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 4.5],
      [2, 3.5],
      [1.5, 5]
    ]

    let logreg = new LogisticRegression({ penalty: 'none' })
    await logreg.fit(X, y)
    let results = logreg.predict(Xtest) // compute results of the training set
    assert.deepEqual(results.arraySync(), arrayTo2DColumn([0, 0, 0, 1, 1, 1]))
  })
  it('Test of the prediction with 3 classes', async function () {
    let X = [
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
      [1, 10],
      [1, 12],
      [2, 10],
      [2, 11],
      [2, 14],
      [3, 11]
    ]
    let y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    let Xtest = [
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5]
    ]

    let logreg = new LogisticRegression({ penalty: 'none' })
    await logreg.fit(X, y)
    let finalResults = logreg.predict(Xtest)
    console.log(finalResults.arraySync())
    assert.deepEqual(
      finalResults.arraySync(),
      arrayTo2DColumn([0, 0, 0, 1, 1, 1, 2, 2, 2])
    )
  })
  // it('Works on small multi-output example (small example)', async function () {
  //   const lr = new LogisticRegression()
  //   await lr.fit(
  //     [[1], [2]],
  //     [
  //       [2, 1],
  //       [4, 2]
  //     ]
  //   )

  //   assert.isTrue(tensorEqual(lr.coef_, tf.tensor2d([[2, 1]]), 0.1))
  // })

  // it('Works on arrays with no intercept (small example)', async function () {
  //   const lr = new LogisticRegression({ fitIntercept: false })
  //   await lr.fit([[1], [2]], [2, 4])
  //   assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2]), 0.1))
  //   assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  // })

  // it('Works on arrays with none zero intercept (small example)', async function () {
  //   const lr = new LogisticRegression({ fitIntercept: true })
  //   await lr.fit([[1], [2]], [3, 5])
  //   assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2]), 0.1))
  //   assert.isTrue(roughlyEqual(lr.intercept_ as number, 1))
  // })

  // // Medium sized example
  // it('Works on arrays with none zero intercept (medium example)', async function () {
  //   const sizeOfMatrix = 100
  //   const seed = 42
  //   let mediumX = tf.randomUniform(
  //     [sizeOfMatrix, 2],
  //     -10,
  //     10,
  //     'float32',
  //     seed
  //   ) as Tensor2D
  //   let [firstCol, secondCol] = mediumX.split([1, 1], 1)
  //   let y = firstCol
  //     .mul(2.5)
  //     .add(secondCol)
  //     .reshape([sizeOfMatrix]) as Tensor1D
  //   const yPlusJitter = y.add(
  //     tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
  //   ) as Tensor1D
  //   const lr = new LogisticRegression({ fitIntercept: false })
  //   await lr.fit(mediumX, yPlusJitter)

  //   assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2.5, 1]), 0.1))
  //   assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  // })

  // it('Works on arrays with none zero intercept (medium example)', async function () {
  //   const sizeOfMatrix = 1000
  //   const seed = 42
  //   let mediumX = tf.randomUniform(
  //     [sizeOfMatrix, 2],
  //     -10,
  //     10,
  //     'float32',
  //     seed
  //   ) as Tensor2D
  //   let [firstCol, secondCol] = mediumX.split([1, 1], 1)
  //   let y = firstCol
  //     .mul(2.5)
  //     .add(secondCol)
  //     .reshape([sizeOfMatrix]) as Tensor1D
  //   const yPlusJitter = y.add(
  //     tf.randomNormal([sizeOfMatrix], 0, 1, 'float32', seed)
  //   ) as Tensor1D
  //   const lr = new LogisticRegression({ fitIntercept: false })
  //   await lr.fit(mediumX, yPlusJitter)

  //   assert.isTrue(tensorEqual(lr.coef_, tf.tensor1d([2.5, 1]), 0.1))
  //   assert.isTrue(roughlyEqual(lr.intercept_ as number, 0))
  // })
})
