import { assert } from 'chai'
import { VotingRegressor } from './votingRegressor'
import { LinearRegression } from '../estimators/linearRegression'
import { DummyRegressor } from '../dummy/dummyRegressor'
import { describe, it } from 'mocha'

describe('VotingRegressor', function () {
  it('Use VotingRegressor on simple example ', async function () {
    const X = [
      [2, 2],
      [2, 3],
      [5, 4],
      [1, 0]
    ]
    const y = [5, 3, 4, 1.5]
    const voter = new VotingRegressor({
      estimators: [
        ['dt', new DummyRegressor()],
        ['lr', new LinearRegression({ fitIntercept: false })]
      ]
    })

    await voter.fit(X, y)

    assert.isTrue(voter.score(X, y) > 0.9)
  })
  // it('Use VotingRegressor on simple example (median)', function () {
  //   const reg = new VotingRegressor({ strategy: 'median' })

  //   const X = [
  //     [-1, 5],
  //     [-0.5, 5],
  //     [0, 10]
  //   ]
  //   const y = [10, 12, 30]
  //   const predictX = [
  //     [1, 0],
  //     [1, 1],
  //     [1, 1]
  //   ]

  //   reg.fit(X, y)
  //   assert.deepEqual(reg.predict(predictX).arraySync(), [12, 12, 12])
  // })
  // it('Use VotingRegressor on simple example (constant)', function () {
  //   const reg = new VotingRegressor({ strategy: 'constant', constant: 10 })

  //   const X = [
  //     [-1, 5],
  //     [-0.5, 5],
  //     [0, 10]
  //   ]
  //   const y = [10, 12, 30]
  //   const predictX = [
  //     [1, 0],
  //     [1, 1],
  //     [1, 1]
  //   ]

  //   reg.fit(X, y)
  //   assert.deepEqual(reg.predict(predictX).arraySync(), [10, 10, 10])
  // })
})
