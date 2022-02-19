import '@tensorflow/tfjs-backend-webgl'
import { assert } from 'chai'
import { LinearSVC } from './linearSVC'
import { describe, it } from 'mocha'

describe('LinearSVC', function () {
  this.timeout(30000)
  it('Works on arrays (small example)', async function () {
    const lr = new LinearSVC()

    await lr.fit([[1], [2]], [0, 1])
    assert.deepEqual(lr.predict([[1], [2]]).arraySync(), [0, 1])
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

    let svc = new LinearSVC({ penalty: 'none' })
    await svc.fit(X, y)
    let results = svc.predict(Xtest) // compute results of the training set
    assert.deepEqual(results.arraySync(), [0, 0, 0, 1, 1, 1])
    assert.isTrue(svc.score(X, y) > 0.5)
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

    let svc = new LinearSVC({ penalty: 'none' })
    await svc.fit(X, y)
    let finalResults = svc.predict(Xtest)
    assert.deepEqual(finalResults.arraySync(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
  })
})
