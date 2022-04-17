import {
  ClassificationCriterion,
  RegressionCriterion,
  ImpurityMeasure
} from './criterion'
import { shuffle } from 'lodash'
import { int } from '../randUtils'
import Serialize from '../serialize'

export interface Split {
  feature: int
  threshold: int
  pos: int
  impurityLeft: number
  impurityRight: number
  foundSplit: boolean
}

export function makeDefaultSplit() {
  return {
    feature: 0,
    threshold: 0,
    pos: -1,
    impurityLeft: Number.POSITIVE_INFINITY,
    impurityRight: Number.POSITIVE_INFINITY,
    foundSplit: false
  }
}

export class Splitter extends Serialize {
  kMinSplitDiff: number
  X: number[][]
  y: int[]
  criterion: ClassificationCriterion | RegressionCriterion
  start: int
  end: int
  minSamplesLeaf: int
  maxFeatures: int
  featureOrder: int[]
  shuffleFeatures: boolean
  sampleMap: Int32Array
  nSamplesTotal: int
  nFeatures: int
  name = 'splitter'

  constructor(
    X: number[][],
    y: int[],
    minSamplesLeaf: int,
    impurityMeasure: ImpurityMeasure,
    maxFeatures: int,
    samplesSubset: int[] = []
  ) {
    super()
    this.X = X
    this.y = y
    this.nFeatures = X[0].length
    this.minSamplesLeaf = minSamplesLeaf
    this.maxFeatures = Math.min(maxFeatures, this.nFeatures)
    this.shuffleFeatures = maxFeatures < this.nFeatures
    this.sampleMap = new Int32Array(X.length)
    this.start = 0
    this.end = 0
    this.kMinSplitDiff = 1e-8
    if (samplesSubset.length === 0) {
      this.nSamplesTotal = X.length
      for (let i = 0; i < this.nSamplesTotal; i++) {
        this.sampleMap[i] = i
      }
    } else {
      this.nSamplesTotal = samplesSubset.length
      for (let i = 0; i < this.nSamplesTotal; i++) {
        this.sampleMap[i] = samplesSubset[i]
      }
    }
    if (impurityMeasure === 'squared_error') {
      this.criterion = new RegressionCriterion(impurityMeasure, y)
    } else {
      this.criterion = new ClassificationCriterion(impurityMeasure, y)
    }
    this.featureOrder = []
    for (let i = 0; i < this.nFeatures; i++) {
      this.featureOrder.push(i)
    }
    this.resetSampleRange(0, this.nSamplesTotal)
  }

  resetSampleRange(start: int, end: int) {
    this.start = start
    this.end = end
    this.criterion.init(start, end, this.sampleMap)
  }

  splitNode(): Split {
    let currentSplit = makeDefaultSplit()
    let bestSplit = makeDefaultSplit()
    let currentImpurityImprovement = Number.NEGATIVE_INFINITY
    let bestImpurityImprovement = Number.NEGATIVE_INFINITY
    let currentFeatureNum = 0
    let currentFeature = 0
    currentSplit.foundSplit = false
    if (this.shuffleFeatures) {
      this.featureOrder = shuffle(this.featureOrder)
    }

    while (currentFeatureNum < this.maxFeatures) {
      currentFeature = this.featureOrder[currentFeatureNum]
      let currentFeatureValues = new Float32Array(this.end - this.start)
      for (let i = this.start; i < this.end; i++) {
        let row = this.X[this.sampleMap[i]]
        let val = row[currentFeature]
        currentFeatureValues[i - this.start] = val
      }
      currentFeatureValues.sort()
      this.criterion.reset()

      this.sampleMap
        .subarray(this.start, this.end)
        .sort((a, b) => this.X[a][currentFeature] - this.X[b][currentFeature])

      // If this feature value is constant, then skip it.
      if (
        currentFeatureValues[0] ===
        currentFeatureValues[currentFeatureValues.length - 1]
      ) {
        currentFeatureNum += 1
        continue
      }
      let pos = this.start + 1
      // Loop over all split points
      while (pos < this.end) {
        // Skip split points where the features are equal because
        // you can't "slice" there
        while (
          pos < this.end &&
          currentFeatureValues[pos - this.start] <=
            currentFeatureValues[pos - this.start - 1] + this.kMinSplitDiff
        ) {
          pos++
        }
        if (pos === this.end) {
          pos++
          continue
        }
        // Check if split would lead to less than minSamplesLeaf samples
        if (
          !(
            pos - this.start < this.minSamplesLeaf ||
            this.end - pos < this.minSamplesLeaf
          )
        ) {
          currentSplit.pos = pos
          this.criterion.update(currentSplit.pos, this.sampleMap)
          currentImpurityImprovement = this.criterion.impurityImprovement()
          if (currentImpurityImprovement > bestImpurityImprovement) {
            bestImpurityImprovement = currentImpurityImprovement
            currentSplit.foundSplit = true
            currentSplit.feature = currentFeature

            currentSplit.threshold =
              (currentFeatureValues[pos - this.start - 1] +
                currentFeatureValues[pos - this.start]) /
              2.0

            bestSplit = Object.assign({}, currentSplit)
          }
        }

        // increment the position
        pos += 1
      }
      // increment the feature that we are looking at
      currentFeatureNum += 1
    }

    if (currentSplit.foundSplit) {
      if (bestSplit.pos < this.end) {
        if (currentFeature !== bestSplit.feature) {
          let leftPos = this.start
          let rightPos = this.end
          let tmp = 0
          while (leftPos < rightPos) {
            if (
              this.X[this.sampleMap[leftPos]][bestSplit.feature] <=
              bestSplit.threshold
            ) {
              leftPos += 1
            } else {
              rightPos -= 1
              tmp = this.sampleMap[leftPos]
              this.sampleMap[leftPos] = this.sampleMap[rightPos]
              this.sampleMap[rightPos] = tmp
            }
          }
        }
      }

      this.criterion.reset()
      this.criterion.update(bestSplit.pos, this.sampleMap)
      let { impurityLeft, impurityRight } = this.criterion.childrenImpurities()

      bestSplit.impurityLeft = impurityLeft
      bestSplit.impurityRight = impurityRight

      return bestSplit
    } else {
      // passing back split.foundSplit = false
      return currentSplit
    }
  }

  public toJson(): string {
    const jsonClass = JSON.parse(super.toJson() as string)

    if (jsonClass.criterion) {
      jsonClass.criterion = this.criterion.toJson() as string
    }
    if (this.sampleMap) jsonClass.sampleMap = Array.from(this.sampleMap)
    return JSON.stringify(jsonClass)
  }

  static fromJson(model: string) {
    const jsonClass = JSON.parse(model)

    if (jsonClass.criterion) {
      const criterionName = JSON.parse(jsonClass.criterion).name
      if (criterionName == 'classificationCriterion') {
        jsonClass.criterion = ClassificationCriterion.fromJson(
          jsonClass.criterion
        )
      } else {
        jsonClass.criterion = RegressionCriterion.fromJson(jsonClass.criterion)
      }
    }

    if (jsonClass.sampleMap) {
      jsonClass.sampleMap = new Int32Array(jsonClass.sampleMap)
    }

    const splitter = new Splitter(
      jsonClass.X,
      jsonClass.y,
      jsonClass.minSamplesLeaf,
      'squared_error',
      jsonClass.samplesSubset
    )

    return Object.assign(splitter, jsonClass) as Splitter
  }
}
