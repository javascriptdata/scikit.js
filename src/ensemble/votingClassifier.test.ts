import { assert } from 'chai'
import { makeVotingClassifier, VotingClassifier } from './votingClassifier'
import { DummyClassifier } from '../dummy/dummyClassifier'
import { describe, it } from 'mocha'
import { LogisticRegression } from '../linear_model/logisticRegression'

describe('VotingClassifier', function () {
  this.timeout(10000)
  it('Use VotingClassifier on simple example (voting = hard)', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1],
      [4, 4]
    ]
    const y = [0, 0, 1, 1, 1]
    const voter = new VotingClassifier({
      estimators: [
        ['dt', new DummyClassifier()],
        ['dt', new DummyClassifier()],
        ['lr', new LogisticRegression({ penalty: 'none' })]
      ]
    })

    await voter.fit(X, y)
    assert.deepEqual(voter.predict(X).arraySync(), [1, 1, 1, 1, 1])
  })
  it('Use VotingClassifier on simple example label encoder (voting = hard)', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1],
      [4, 4]
    ]
    const y = [1, 1, 2, 2, 2]
    const voter = new VotingClassifier({
      estimators: [
        ['dt', new DummyClassifier()],
        ['dt', new DummyClassifier()],
        ['lr', new LogisticRegression({ penalty: 'none' })]
      ]
    })

    await voter.fit(X, y)
    assert.deepEqual(voter.predict(X).arraySync(), [2, 2, 2, 2, 2])
  })
  it('Use VotingClassifier on simple example label encoder (voting = soft)', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1],
      [4, 4]
    ]
    const y = [0, 0, 1, 1, 1]
    const voter = new VotingClassifier({
      estimators: [
        ['dt', new DummyClassifier()],
        ['lr', new LogisticRegression({ penalty: 'none' })]
      ],
      voting: 'soft'
    })

    await voter.fit(X, y)
    assert.deepEqual(voter.predict(X).arraySync(), [1, 1, 1, 1, 1])
  })
  it('Use VotingClassifier on simple example label encoder (voting = soft)', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1],
      [4, 4]
    ]
    const y = [0, 0, 1, 1, 1]
    const voter = new VotingClassifier({
      estimators: [
        ['dt', new DummyClassifier()],
        ['lr', new LogisticRegression({ penalty: 'none' })]
      ],
      voting: 'soft',
      weights: [0.1, 0.9]
    })

    await voter.fit(X, y)
    assert.deepEqual(voter.predict(X).arraySync(), [0, 0, 1, 1, 1])
  })
  it('Use VotingClassifier on simple example label encoder (voting = soft)', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1],
      [4, 4]
    ]
    const y = [0, 0, 1, 1, 1]
    const voter = makeVotingClassifier(
      new DummyClassifier(),
      new DummyClassifier()
    )

    await voter.fit(X, y)
    assert.deepEqual(voter.predict(X).arraySync(), [1, 1, 1, 1, 1])
  })
})
