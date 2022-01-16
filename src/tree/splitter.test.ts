import { ImpurityMeasure } from './criterion'
import { Splitter } from './splitter'

describe('Splitter', function () {
  let types = ['gini', 'entropy', 'mse']
  it('Use the criterion (init)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [0, 0, 0, 1, 1, 1]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 1, type as ImpurityMeasure, 1, [])

      let best_split = splitter.splitNode()
      expect(best_split.threshold).toEqual(0)
      expect(best_split.feature).toEqual(0)
      expect(best_split.pos).toEqual(3)
    })
  }, 1000)
  it('Use the criterion (init diff example)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [1, 1, 0, 1, 1, 1]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 1, type as ImpurityMeasure, 1, [])

      let best_split = splitter.splitNode()
      expect(best_split.threshold).toEqual(0)
      expect(best_split.feature).toEqual(0)
      expect(best_split.pos).toEqual(3)
    })
  }, 1000)
  it('Use the criterion (init diff example 2)', async function () {
    let X = [[-2], [-1], [0], [1], [1], [2]]
    let y = [1, 0, 1, 1, 1, 1]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 1, type as ImpurityMeasure, 1, [])
      let best_split = splitter.splitNode()
      expect(best_split.threshold).toEqual(-0.5)
      expect(best_split.feature).toEqual(0)
      expect(best_split.pos).toEqual(2)
    })
  }, 1000)

  it('Use the criterion (init diff example mse)', async function () {
    let X = [[1], [1], [1], [1], [1], [1], [1], [1]]
    let y = [1, 1, 1, 1, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 1, type as ImpurityMeasure, 1, [])
      let best_split = splitter.splitNode()
      expect(best_split.found_split).toEqual(false)
      expect(best_split.threshold).toEqual(0)
      expect(best_split.feature).toEqual(0)
      expect(best_split.pos).toEqual(-1)
    })
  }, 1000)
  it('Use the criterion (init diff example min samples test)', async function () {
    let X = [[0], [1], [2], [3], [4], [5], [6], [7]]
    let y = [1, 1, 1, 2, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 4, type as ImpurityMeasure, 1, [])
      let best_split = splitter.splitNode()
      expect(best_split.found_split).toEqual(true)
      expect(best_split.feature).toEqual(0)
      expect(best_split.threshold).toEqual(3.5)
      expect(best_split.pos).toEqual(4)
    })
  }, 1000)
  it('Use the criterion (init diff example min samples test)', async function () {
    let X = [[0], [1], [2], [3], [4], [5], [6], [7]]
    let y = [1, 1, 1, 2, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 4, type as ImpurityMeasure, 1, [])
      let best_split = splitter.splitNode()
      expect(best_split.found_split).toEqual(true)
      expect(best_split.feature).toEqual(0)
      expect(best_split.threshold).toEqual(3.5)
      expect(best_split.pos).toEqual(4)
    })
  }, 1000)
  it('Use the criterion (init diff example min samples test)', async function () {
    let X = [
      [0, 1],
      [1, 1],
      [1, 2],
      [2, 2],
      [2, 3],
      [2, 3],
      [3, 4],
      [3, 4]
    ]
    let y = [1, 1, 1, 1, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 1, type as ImpurityMeasure, 20, [])
      let best_split = splitter.splitNode()
      expect(best_split.found_split).toEqual(true)
      expect(best_split.feature).toEqual(1)
      expect(best_split.threshold).toEqual(2.5)
      expect(best_split.pos).toEqual(4)
    })
  }, 1000)
  it('Use the criterion (init diff example min samples test)', async function () {
    let X = [
      [3, 4],
      [1, 1],
      [1, 2],
      [2, 3],
      [2, 2],
      [2, 3],
      [3, 4],
      [0, 1]
    ]
    let y = [2, 1, 1, 2, 1, 2, 2, 1]

    types.forEach((type) => {
      let splitter = new Splitter(X, y, 1, type as ImpurityMeasure, 20, [])
      let best_split = splitter.splitNode()
      expect(best_split.found_split).toEqual(true)
      expect(best_split.feature).toEqual(1)
      expect(best_split.threshold).toEqual(2.5)
      expect(best_split.pos).toEqual(4)
    })
  }, 1000)
})
