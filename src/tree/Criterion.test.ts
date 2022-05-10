import { ClassificationCriterion, giniCoefficient, entropy } from './Criterion'
import { fromJSON } from '../simpleSerializer'
import { setBackend } from '../index'
import * as tf from '@tensorflow/tfjs'
setBackend(tf)
describe('Criterion', function () {
  let X = [
    [-2, -1],
    [-1, -1],
    [-1, -2],
    [1, 1],
    [1, 2],
    [2, 1]
  ]
  let y = [0, 0, 0, 1, 1, 1]
  let sampleMap = new Int32Array(X.length)
  for (let i = 0; i < X.length; i++) {
    sampleMap[i] = i
  }
  it('Use the criterion (init)', async function () {
    let criterion = new ClassificationCriterion({ impurityMeasure: 'gini', y })

    criterion.init(0, 6, sampleMap)
    expect(criterion.start).toEqual(0)
    expect(criterion.end).toEqual(6)
    expect(criterion.labelFreqsTotal[0]).toEqual(3)
    expect(criterion.labelFreqsTotal[1]).toEqual(3)

    expect(criterion.labelFreqsLeft[0]).toEqual(0)
    expect(criterion.labelFreqsLeft[1]).toEqual(0)
    expect(criterion.labelFreqsRight[0]).toEqual(0)
    expect(criterion.labelFreqsRight[1]).toEqual(0)
  }, 1000)
  it('Use the criterion (update)', async function () {
    let criterion = new ClassificationCriterion({ impurityMeasure: 'gini', y })
    criterion.init(0, 6, sampleMap)
    criterion.update(3, sampleMap)

    expect(criterion.pos).toEqual(3)
    expect(criterion.labelFreqsLeft[0]).toEqual(3)
    expect(criterion.labelFreqsLeft[1]).toEqual(0)
    expect(criterion.labelFreqsRight[0]).toEqual(0)
    expect(criterion.labelFreqsRight[1]).toEqual(3)
  }, 1000)
  it('Use the criterion (gini)', async function () {
    let criterion = new ClassificationCriterion({ impurityMeasure: 'gini', y })

    criterion.init(0, 6, sampleMap)

    expect(criterion.nodeImpurity()).toEqual(0.5)
  }, 1000)
  it('Use the criterion (entropy)', async function () {
    let criterion = new ClassificationCriterion({
      impurityMeasure: 'entropy',
      y
    })
    criterion.init(0, 6, sampleMap)

    expect(criterion.nodeImpurity()).toEqual(1)
  }, 1000)
  it('Use the criterion (gini update)', async function () {
    let criterion = new ClassificationCriterion({ impurityMeasure: 'gini', y })

    criterion.init(0, 6, sampleMap)
    criterion.update(4, sampleMap)

    expect(criterion.impurityImprovement()).toEqual(-1.5)

    let { impurityLeft, impurityRight } = criterion.childrenImpurities()
    expect(impurityLeft).toEqual(0.375)
    expect(impurityRight).toEqual(0)
  }, 1000)
  it('Gini coef', async function () {
    let labelFreqs = [20, 80]
    let nSamples = 100
    expect(giniCoefficient(labelFreqs, nSamples)).toEqual(0.31999999999999995)
  }, 1000)
  it('Entropy coef', async function () {
    let labelFreqs = [20, 80]
    let nSamples = 100
    expect(entropy(labelFreqs, nSamples)).toEqual(0.7219280948873623)
  }, 1000)
  it('Use the criterion (entropy)', async function () {
    let criterion = new ClassificationCriterion({
      impurityMeasure: 'entropy',
      y
    })
    criterion.init(0, 6, sampleMap)
    const serial = await criterion.toJSON()
    const newCriterion = await fromJSON(serial)
    expect(newCriterion.nodeImpurity()).toEqual(1)
  }, 1000)
})
