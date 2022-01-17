import {
  ClassificationCriterion,
  RegressionCriterion,
  ImpurityMeasure,
  SampleData
} from './criterion'
import { shuffle } from 'lodash'
import { quickSort } from './utils'
import { int } from '../randUtils'

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

export class Splitter {
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
  sampleMap: SampleData[]
  nSamplesTotal: int
  nFeatures: int

  constructor(
    X: number[][],
    y: int[],
    minSamplesLeaf: int,
    impurityMeasure: ImpurityMeasure,
    maxFeatures: int,
    samplesSubset: int[] = []
  ) {
    this.X = X
    this.y = y
    this.nFeatures = X[0].length
    this.minSamplesLeaf = minSamplesLeaf
    this.maxFeatures = maxFeatures
    this.shuffleFeatures = maxFeatures < this.nFeatures
    this.sampleMap = []
    this.start = 0
    this.end = 0
    this.kMinSplitDiff = 1e-8
    if (samplesSubset.length === 0) {
      this.nSamplesTotal = X.length
      for (let i = 0; i < this.nSamplesTotal; i++) {
        this.sampleMap.push({ currentFeatureValue: 0, sampleNumber: i })
      }
    } else {
      this.nSamplesTotal = samplesSubset.length
      for (let i = 0; i < this.nSamplesTotal; i++) {
        this.sampleMap.push({
          currentFeatureValue: 0,
          sampleNumber: samplesSubset[i]
        })
      }
    }
    if (impurityMeasure === 'mse') {
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

      // Copies feature data into sample map
      for (let i = this.start; i < this.end; i++) {
        this.sampleMap[i].currentFeatureValue =
          this.X[this.sampleMap[i].sampleNumber][currentFeature]
      }
      this.criterion.reset()
      this.sampleMap = quickSort(
        this.sampleMap,
        this.start,
        this.end - 1,
        'currentFeatureValue'
      )

      // If this feature value is constant, then skip it.
      if (
        this.sampleMap[this.start].currentFeatureValue ===
        this.sampleMap[this.end - 1].currentFeatureValue
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
          this.sampleMap[pos].currentFeatureValue <=
            this.sampleMap[pos - 1].currentFeatureValue + this.kMinSplitDiff
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
              (this.sampleMap[pos - 1].currentFeatureValue +
                this.sampleMap[pos].currentFeatureValue) /
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
              this.X[this.sampleMap[leftPos].sampleNumber][
                bestSplit.feature
              ] <= bestSplit.threshold
            ) {
              leftPos += 1
            } else {
              rightPos -= 1
              tmp = this.sampleMap[leftPos].sampleNumber
              this.sampleMap[leftPos].sampleNumber =
                this.sampleMap[rightPos].sampleNumber
              this.sampleMap[rightPos].sampleNumber = tmp
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
}
