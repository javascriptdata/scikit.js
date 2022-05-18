import {
  makeVotingRegressor,
  VotingRegressor,
  DummyRegressor,
  LinearRegression,
  setBackend,
  fromJSON
} from '../index'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

describe('VotingRegressor', function () {
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
    expect(voter.score(X, y) > 0).toBe(true)
  }, 30000)
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
    expect(voter.score(X, y) > 0).toBe(true)
  }, 30000)
  it('Should save and load VotingRegressor ', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1]
    ]
    const y = [3, 3, 4, 4]
    const voter = makeVotingRegressor(
      new LinearRegression({ fitIntercept: true })
    )

    await voter.fit(X, y)

    const savedModel = await voter.toJSON()
    const newModel = await fromJSON(savedModel)
    expect(newModel.score(X, y)).toEqual(voter.score(X, y))
  }, 30000)
})
