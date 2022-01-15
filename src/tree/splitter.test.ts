import { Splitter } from './splitter'

describe('Splitter', function () {
  it('Use the criterion (init)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [0, 0, 0, 1, 1, 1]

    let splitter = new Splitter(X, y, 1, 'gini', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(0)
    expect(best_split.pos).toEqual(3)
  }, 1000)
  it('Use the criterion (init diff example)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [1, 1, 0, 1, 1, 1]

    let splitter = new Splitter(X, y, 1, 'gini', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(0)
    expect(best_split.pos).toEqual(3)
  }, 1000)
  it('Use the criterion (init diff example 2)', async function () {
    let X = [[-2], [-1], [0], [1], [1], [2]]
    let y = [1, 0, 1, 1, 1, 1]

    let splitter = new Splitter(X, y, 1, 'gini', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(-0.5)
    expect(best_split.pos).toEqual(2)
  }, 1000)
  it('Use the criterion (init diff example mse)', async function () {
    let X = [[1], [2], [3], [4], [5], [6], [7], [8]]
    let y = [1, 1, 1, 1, 2, 2, 2, 2]

    let splitter = new Splitter(X, y, 1, 'mse', 1, [])

    let best_split = splitter.splitNode()
    expect(best_split.threshold).toEqual(4.5)
    expect(best_split.pos).toEqual(4)
  }, 1000)
})
