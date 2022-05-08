import { int } from '../randUtils'
import { Serialize } from '../simpleSerializer'

export type ImpurityMeasure = 'gini' | 'entropy' | 'squared_error'

export function giniCoefficient(labelFreqs: int[], nSamples: int) {
  let freqSquares = 0
  for (let i = 0; i < labelFreqs.length; i++) {
    freqSquares += labelFreqs[i] * labelFreqs[i]
  }
  return 1 - freqSquares / (nSamples * nSamples)
}

export function entropy(labelFreqs: int[], nSamples: int) {
  let totalEntropy = 0
  for (let i = 0; i < labelFreqs.length; i++) {
    let labelFrequency = labelFreqs[i]
    if (labelFrequency > 0) {
      labelFrequency /= nSamples
      totalEntropy -= labelFrequency * Math.log2(labelFrequency)
    }
  }
  return totalEntropy
}

export function mse(ySquaredSum: number, ySum: number, nSamples: int) {
  let yBar = ySum / nSamples
  let val = ySquaredSum / nSamples - yBar * yBar
  return val
}

function arrayMax(labels: int[]) {
  let max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] > max) {
      max = labels[i]
    }
  }
  return max
}

export class ClassificationCriterion extends Serialize {
  y: int[]
  impurityMeasure: ImpurityMeasure
  // impurityFunc: (labelFreqs: int[], nSamples: int) => number
  start: int = 0
  end: int = 0
  pos: int = 0
  nLabels: int
  labelFreqsTotal: int[] = []
  labelFreqsLeft: int[] = []
  labelFreqsRight: int[] = []
  nSamples: int = 0
  nSamplesLeft: int = 0
  nSamplesRight: int = 0
  name = 'ClassificationCriterion'

  constructor({
    impurityMeasure,
    y
  }: {
    impurityMeasure: ImpurityMeasure
    y: number[]
  }) {
    super()

    this.impurityMeasure = impurityMeasure

    // This assumes that the labels are 0,1,2,...,(n-1)
    this.nLabels = arrayMax(y) + 1
    this.y = y
    this.labelFreqsTotal = new Array(this.nLabels).fill(0)
    this.labelFreqsLeft = new Array(this.nLabels).fill(0)
    this.labelFreqsRight = new Array(this.nLabels).fill(0)
  }

  init(start: int, end: int, sampleMap: Int32Array) {
    this.start = start
    this.end = end
    this.nSamples = end - start
    this.labelFreqsTotal = this.labelFreqsTotal.fill(0)
    this.labelFreqsLeft = this.labelFreqsLeft.fill(0)
    this.labelFreqsRight = this.labelFreqsRight.fill(0)

    for (let i = start; i < end; i++) {
      let sampleNumber = sampleMap[i]
      this.labelFreqsTotal[this.y[sampleNumber]] += 1
    }
  }

  reset() {
    this.pos = this.start
    this.labelFreqsLeft = this.labelFreqsLeft.fill(0)
    this.labelFreqsRight = this.labelFreqsRight.fill(0)
  }

  update(newPos: int, sampleMap: Int32Array) {
    for (let i = this.pos; i < newPos; i++) {
      // This assumes that the labels take values 0,..., nLabels - 1
      let sampleNumber = sampleMap[i]
      this.labelFreqsLeft[this.y[sampleNumber]] += 1
    }

    // calculate labelFreqsRight
    for (let i = 0; i < this.labelFreqsTotal.length; i++) {
      this.labelFreqsRight[i] =
        this.labelFreqsTotal[i] - this.labelFreqsLeft[i]
    }

    this.pos = newPos
    this.nSamplesLeft = this.pos - this.start
    this.nSamplesRight = this.end - this.pos
  }

  childrenImpurities() {
    let impurityFunc =
      this.impurityMeasure === 'gini' ? giniCoefficient : entropy

    return {
      impurityLeft: impurityFunc(this.labelFreqsLeft, this.nSamplesLeft),
      impurityRight: impurityFunc(this.labelFreqsRight, this.nSamplesRight)
    }
  }

  impurityImprovement() {
    let { impurityLeft, impurityRight } = this.childrenImpurities()

    return (
      -this.nSamplesLeft * impurityLeft - this.nSamplesRight * impurityRight
    )
  }

  nodeImpurity() {
    let impurityFunc =
      this.impurityMeasure === 'gini' ? giniCoefficient : entropy

    return impurityFunc(this.labelFreqsTotal, this.nSamples)
  }

  nodeValue() {
    return this.labelFreqsTotal
  }

  static fromJson(model: string) {
    const jsonClass = JSON.parse(model)
    const newModel = new ClassificationCriterion({
      impurityMeasure: jsonClass.impurityMeasure,
      y: jsonClass.y
    })
    return Object.assign(newModel, jsonClass)
  }
}

export class RegressionCriterion extends Serialize {
  y: number[]
  impurityMeasure: 'squared_error'
  // impurityFunc: (ySquaredSum: number, ySum: number, nSamples: int) => number
  start: int = 0
  end: int = 0
  pos: int = 0
  squaredSum = 0
  squaredSumLeft = 0
  squaredSumRight = 0
  sumTotal = 0
  sumTotalLeft = 0
  sumTotalRight = 0
  nSamples: int = 0
  nSamplesLeft: int = 0
  nSamplesRight: int = 0
  name = 'RegressionCriterion'

  constructor({
    impurityMeasure,
    y
  }: {
    impurityMeasure: 'squared_error'
    y: number[]
  }) {
    super()

    // We don't assert in the constructor, we assert in fit in accordance with the sklearn docs

    // Support MAE one day
    this.impurityMeasure = impurityMeasure
    // We don't set the impurityFunc here because we need it to be serializable as an object
    // this.impurityFunc = mse
    this.y = y
  }

  init(start: int, end: int, sampleMap: Int32Array) {
    this.sumTotal = 0
    this.squaredSum = 0
    this.start = start
    this.end = end
    this.nSamples = end - start

    for (let i = start; i < end; i++) {
      let sampleNumber = sampleMap[i]
      let yValue = this.y[sampleNumber]
      this.sumTotal += yValue
      this.squaredSum += yValue * yValue
    }
  }

  reset() {
    this.pos = this.start
    this.squaredSumLeft = 0
    this.sumTotalLeft = 0
    this.squaredSumRight = 0
    this.sumTotalRight = 0
  }

  update(newPos: int, sampleMap: Int32Array) {
    for (let i = this.pos; i < newPos; i++) {
      // This assumes that the labels take values 0,..., nLabels - 1
      let sampleNumber = sampleMap[i]
      let yValue = this.y[sampleNumber]
      this.sumTotalLeft += yValue
      this.squaredSumLeft += yValue * yValue
    }

    // calculate labelFreqsRight
    this.sumTotalRight = this.sumTotal - this.sumTotalLeft
    this.squaredSumRight = this.squaredSum - this.squaredSumLeft

    this.pos = newPos
    this.nSamplesLeft = this.pos - this.start
    this.nSamplesRight = this.end - this.pos
  }

  childrenImpurities() {
    // once we get another impurity function we can do a ternary here
    let impurityFunc = mse
    return {
      impurityLeft: impurityFunc(
        this.squaredSumLeft,
        this.sumTotalLeft,
        this.nSamplesLeft
      ),
      impurityRight: impurityFunc(
        this.squaredSumRight,
        this.sumTotalRight,
        this.nSamplesRight
      )
    }
  }

  impurityImprovement() {
    let { impurityLeft, impurityRight } = this.childrenImpurities()

    return (
      -this.nSamplesLeft * impurityLeft - this.nSamplesRight * impurityRight
    )
  }

  nodeImpurity() {
    // once we get another impurity function we can do a ternary here
    let impurityFunc = mse
    return impurityFunc(this.squaredSum, this.sumTotal, this.nSamples)
  }

  nodeValue() {
    return [this.sumTotal / this.nSamples]
  }

  static fromJson(model: string) {
    const jsonClass = JSON.parse(model)
    const newModel = new RegressionCriterion({
      impurityMeasure: jsonClass.impurityMeasure,
      y: jsonClass.y
    })
    return Object.assign(newModel, jsonClass)
  }
}
