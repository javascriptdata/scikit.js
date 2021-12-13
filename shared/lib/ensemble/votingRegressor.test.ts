import { assert } from 'chai'
import { VotingRegressor } from './votingRegressor'
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
    const y = [3, 3, 4, 4]
    const voter = new VotingRegressor({
      estimators: [
        ['dt', new DummyRegressor()],
        ['lr', new DummyRegressor({ strategy: 'median' })]
      ]
    })

    await voter.fit(X, y)
    assert.isTrue(voter.score(X, y) >= 0)
  })
})
