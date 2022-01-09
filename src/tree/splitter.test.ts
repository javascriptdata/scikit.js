import { Splitter } from './splitter'

describe('Splitter', function () {
  it('Use the criterion (init)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [0, 0, 0, 1, 1, 1]

    let splitter = new Splitter(X, y, 1, 'gini', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(0)
    expect(best_split.pos).toEqual(3)
    expect(best_split.left_value).toEqual([3, 0])
    expect(best_split.right_value).toEqual([0, 3])
  }, 1000)
  it('Use the criterion (init diff example)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [1, 1, 0, 1, 1, 1]

    let splitter = new Splitter(X, y, 1, 'gini', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(0)
    expect(best_split.pos).toEqual(3)
    expect(best_split.left_value).toEqual([1, 2])
    expect(best_split.right_value).toEqual([0, 3])
  }, 1000)
  it('Use the criterion (init diff example 2)', async function () {
    let X = [[-2], [-1], [0], [1], [1], [2]]
    let y = [1, 0, 1, 1, 1, 1]

    let splitter = new Splitter(X, y, 1, 'gini', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(-0.5)
    expect(best_split.pos).toEqual(2)
    expect(best_split.left_value).toEqual([1, 1])
    expect(best_split.right_value).toEqual([0, 4])
  }, 1000)
})
