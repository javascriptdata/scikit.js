import { assert } from 'chai'
import { KNeighborsRegressor } from './kNeighborsRegressor'
import { describe, it } from 'mocha'

describe('KNeighborsRegressor', function () {
  it('Use KNeighborsRegressor on simple example (n=1)', async function () {
    const knn = new KNeighborsRegressor({ nNeighbors: 1 })

    const X = [
      [-1, 0],
      [0, 0],
      [5, 0]
    ]
    const y = [10, 20, 30]
    const predictX = [
      [1, 0],
      [4, 0],
      [-5, 0]
    ]

    await knn.fit(X, y)
    assert.deepEqual(knn.predict(predictX).arraySync(), [20, 30, 10])
  })
  it('Use KNeighborsRegressor on simple example (n=2)', async function () {
    const knn = new KNeighborsRegressor({ nNeighbors: 2 })

    const X = [
      [-1, 0],
      [0, 0],
      [5, 0]
    ]
    const y = [10, 20, 30]
    const predictX = [
      [1, 0],
      [4, 0],
      [-5, 0]
    ]

    await knn.fit(X, y)
    assert.deepEqual(knn.predict(predictX).arraySync(), [15, 25, 15])
  })
  it('Use KNeighborsRegressor on simple example (n=3)', async function () {
    const knn = new KNeighborsRegressor({ nNeighbors: 3 })

    const X = [
      [-1, 0],
      [0, 0],
      [5, 0]
    ]
    const y = [10, 20, 30]
    const predictX = [
      [1, 0],
      [4, 0],
      [-5, 0]
    ]

    await knn.fit(X, y)
    assert.deepEqual(knn.predict(predictX).arraySync(), [20, 20, 20])
  })
})
