import { ImpurityMeasure } from './Criterion'
import { Splitter } from './Splitter'
import { fromJSON } from '../simpleSerializer'
import { setBackend } from '../index'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)

describe('Splitter', function () {
  let types = ['gini', 'entropy', 'squared_error']
  it('Use the criterion (init)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [0, 0, 0, 1, 1, 1]

    types.forEach((type) => {
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 1,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 1,
        samplesSubset: []
      })

      let bestSplit = splitter.splitNode()
      expect(bestSplit.threshold).toEqual(0)
      expect(bestSplit.feature).toEqual(0)
      expect(bestSplit.pos).toEqual(3)
    })
  }, 1000)
  it('Use the criterion (init diff example)', async function () {
    let X = [[-2], [-1], [-1], [1], [1], [2]]
    let y = [1, 1, 0, 1, 1, 1]

    types.forEach((type) => {
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 1,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 1,
        samplesSubset: []
      })

      let bestSplit = splitter.splitNode()
      expect(bestSplit.threshold).toEqual(0)
      expect(bestSplit.feature).toEqual(0)
      expect(bestSplit.pos).toEqual(3)
    })
  }, 1000)
  it('Use the criterion (init diff example 2)', async function () {
    let X = [[-2], [-1], [0], [1], [1], [2]]
    let y = [1, 0, 1, 1, 1, 1]

    types.forEach((type) => {
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 1,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 1,
        samplesSubset: []
      })
      let bestSplit = splitter.splitNode()
      expect(bestSplit.threshold).toEqual(-0.5)
      expect(bestSplit.feature).toEqual(0)
      expect(bestSplit.pos).toEqual(2)
    })
  }, 1000)

  it('Use the criterion (init diff example 2)', async function () {
    let X = [[1], [1], [1], [1], [1], [1], [1], [1]]
    let y = [1, 1, 1, 1, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 1,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 1,
        samplesSubset: []
      })
      let bestSplit = splitter.splitNode()
      expect(bestSplit.foundSplit).toEqual(false)
      expect(bestSplit.threshold).toEqual(0)
      expect(bestSplit.feature).toEqual(0)
      expect(bestSplit.pos).toEqual(-1)
    })
  }, 1000)
  it('Use the criterion (init diff example min samples test)', async function () {
    let X = [[0], [1], [2], [3], [4], [5], [6], [7]]
    let y = [1, 1, 1, 2, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 4,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 1,
        samplesSubset: []
      })
      let bestSplit = splitter.splitNode()
      expect(bestSplit.foundSplit).toEqual(true)
      expect(bestSplit.feature).toEqual(0)
      expect(bestSplit.threshold).toEqual(3.5)
      expect(bestSplit.pos).toEqual(4)
    })
  }, 1000)
  it('Use the criterion (init diff example min samples test)', async function () {
    let X = [[0], [1], [2], [3], [4], [5], [6], [7]]
    let y = [1, 1, 1, 2, 2, 2, 2, 2]

    types.forEach((type) => {
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 4,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 1,
        samplesSubset: []
      })
      let bestSplit = splitter.splitNode()
      expect(bestSplit.foundSplit).toEqual(true)
      expect(bestSplit.feature).toEqual(0)
      expect(bestSplit.threshold).toEqual(3.5)
      expect(bestSplit.pos).toEqual(4)
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
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 1,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 20,
        samplesSubset: []
      })
      let bestSplit = splitter.splitNode()
      expect(bestSplit.foundSplit).toEqual(true)
      expect(bestSplit.feature).toEqual(1)
      expect(bestSplit.threshold).toEqual(2.5)
      expect(bestSplit.pos).toEqual(4)
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
      let splitter = new Splitter({
        X,
        y,
        minSamplesLeaf: 1,
        impurityMeasure: type as ImpurityMeasure,
        maxFeatures: 20,
        samplesSubset: []
      })
      let bestSplit = splitter.splitNode()
      expect(bestSplit.foundSplit).toEqual(true)
      expect(bestSplit.feature).toEqual(1)
      expect(bestSplit.threshold).toEqual(2.5)
      expect(bestSplit.pos).toEqual(4)
    })
  }, 1000)
  it('Should save and load Splitter', async function () {
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
    let splitter = new Splitter({
      X,
      y,
      minSamplesLeaf: 1,
      impurityMeasure: 'gini',
      maxFeatures: 20,
      samplesSubset: []
    })
    splitter.splitNode()
    const serial = await splitter.toJSON()
    const newSplitter = await fromJSON(serial)
    const newBestSplitter = newSplitter.splitNode()
    expect(newBestSplitter.foundSplit).toEqual(true)
    expect(newBestSplitter.feature).toEqual(1)
    expect(newBestSplitter.threshold).toEqual(2.5)
    expect(newBestSplitter.pos).toEqual(4)
  }, 1000)
})
