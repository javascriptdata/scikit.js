import { assert } from 'chai'
import { SVC } from './SVC'
import { describe, it } from 'mocha'

describe('SVC', function () {
  this.timeout(10000)
  it('Works on arrays (small example)', async function () {
    const lr = new SVC()
    await lr.fit(
      [
        [1, 2],
        [2, -1]
      ],
      [-1, 1]
    )
    const predict = (
      await lr.predict([
        [1, 2],
        [2, -1]
      ])
    ).arraySync()
    assert.deepEqual(predict, [-1, 1])
  })
})
