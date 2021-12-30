import { assert } from 'chai'
import { makeVotingRegressor, VotingRegressor } from './votingRegressor'
import { DummyRegressor } from '../dummy/dummyRegressor'
import { describe, it } from 'mocha'
import { LinearRegression } from '../linear_model/linearRegression'

describe('VotingRegressor', function () {
  this.timeout(10000)
  it('Use VotingRegressor on simple example ', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1]
    ]
    const y = [3, 3, 4, 4]
    const voter = new VotingRegressor({
      estimators: [
        ['dt', new DummyRegressor()],
        ['lr', new LinearRegression({ fitIntercept: true })]
      ]
    })

    await voter.fit(X, y)
    assert.isTrue(voter.score(X, y) > 0)
  })
  it('Use VotingRegressor on simple example ', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1]
    ]
    const y = [3, 3, 4, 4]
    const voter = makeVotingRegressor(
      new DummyRegressor(),
      new LinearRegression({ fitIntercept: true })
    )

    await voter.fit(X, y)
    assert.isTrue(voter.score(X, y) > 0)
  })
})
