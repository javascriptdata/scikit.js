import { ImpurityMeasure } from './criterion'
import { Splitter } from './splitter'
import { int } from '../randUtils'
import { r2Score, accuracyScore } from '../metrics/metrics'
import { Split, makeDefaultSplit } from './splitter'
import { assert, isScikit1D, isScikit2D } from '../typesUtils'
import { validateX, validateY } from './utils'
import { Scikit1D, Scikit2D } from '../types'
import { convertScikit2DToArray, convertScikit1DToArray } from '../utils'
import { LabelEncoder } from '../preprocessing/labelEncoder'
interface NodeRecord {
  start: int
  end: int
  nSamples: int
  depth: int
  parentId: int
  isLeft: boolean
  impurity: number
}

interface Node {
  parentId: int
  leftChildId: int
  rightChildId: int
  isLeft: boolean
  isLeaf: boolean
  impurity: number
  splitFeature: int
  threshold: number
  nSamples: int
  value: int[]
}

function argMax(array: number[]) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1]
}

class DecisionTree {
  nodes: Node[] = []
  isBuilt = false

  getLeafNodes(X: number[][]): int[] {
    let leafNodeIds: int[] = []
    for (let i = 0; i < X.length; i++) {
      let nodeId = 0
      while (!this.nodes[nodeId].isLeaf) {
        if (
          X[i][this.nodes[nodeId].splitFeature] <= this.nodes[nodeId].threshold
        ) {
          nodeId = this.nodes[nodeId].leftChildId
        } else {
          nodeId = this.nodes[nodeId].rightChildId
        }
      }
      leafNodeIds.push(nodeId)
    }
    return leafNodeIds
  }
  populateChildIds(): void {
    for (let i = 1; i < this.nodes.length; i++) {
      if (this.nodes[i].isLeft) {
        this.nodes[this.nodes[i].parentId].leftChildId = i
      } else {
        this.nodes[this.nodes[i].parentId].rightChildId = i
      }
    }
  }
  predictProba(samples: number[][]): number[][] {
    if (!this.isBuilt) {
      throw new Error(
        'Decision tree must be built with BuildTree method before predictions can be made.'
      )
    }
    let leafNodeIds = this.getLeafNodes(samples)
    let classProbabilities = []

    for (let i = 0; i < leafNodeIds.length; i++) {
      let currentClassProbabilities = []
      let curNodeId = leafNodeIds[i]
      for (let nClass = 0; nClass < this.nodes[0].value.length; nClass++) {
        currentClassProbabilities.push(
          this.nodes[curNodeId].value[nClass] / this.nodes[curNodeId].nSamples
        )
      }
      classProbabilities.push(currentClassProbabilities)
    }
    return classProbabilities
  }
  predictClassification(samples: number[][]): int[] {
    if (!this.isBuilt) {
      throw new Error(
        'Decision tree must be built with BuildTree method before predictions can be made.'
      )
    }
    let leafNodeIds = this.getLeafNodes(samples)
    let classPredictions = []

    for (let nSample = 0; nSample < leafNodeIds.length; nSample++) {
      let curNodeId = leafNodeIds[nSample]
      classPredictions.push(argMax(this.nodes[curNodeId].value))
    }
    return classPredictions
  }
  predictRegression(samples: number[][]): int[] {
    if (!this.isBuilt) {
      throw new Error(
        'Decision tree must be built with BuildTree method before predictions can be made.'
      )
    }
    let leafNodeIds = this.getLeafNodes(samples)
    let classPredictions = []

    for (let nSample = 0; nSample < leafNodeIds.length; nSample++) {
      let curNodeId = leafNodeIds[nSample]
      classPredictions.push(this.nodes[curNodeId].value[0])
    }
    return classPredictions
  }
}

interface DecisionTreeBaseParams {
  criterion?: 'gini' | 'entropy' | 'squared_error'
  maxDepth?: int
  minSamplesSplit?: number
  minSamplesLeaf?: number
  maxFeatures?: number | 'auto' | 'sqrt' | 'log2'
  minImpurityDecrease?: number
}

class DecisionTreeBase {
  splitter!: Splitter
  stack: NodeRecord[] = []
  minSamplesLeaf: int
  maxDepth: int
  minSamplesSplit: int
  minImpurityDecrease: number
  tree: DecisionTree
  criterion: ImpurityMeasure
  maxFeatures?: number | 'log2' | 'sqrt' | 'auto'
  maxFeaturesNumb: int
  X: number[][] = []
  y: number[] = []

  constructor({
    criterion = 'gini',
    maxDepth = Number.POSITIVE_INFINITY,
    minSamplesSplit = 2,
    minSamplesLeaf = 1,
    maxFeatures = undefined,
    minImpurityDecrease = 0.0
  }: DecisionTreeBaseParams = {}) {
    this.criterion = criterion as any
    this.maxDepth =
      maxDepth === undefined ? Number.POSITIVE_INFINITY : Number(maxDepth)
    this.minSamplesSplit = minSamplesSplit
    this.minSamplesLeaf = minSamplesLeaf
    this.maxFeatures = maxFeatures
    this.minImpurityDecrease = minImpurityDecrease
    this.maxFeaturesNumb = 0
    this.tree = new DecisionTree()
  }
  calcMaxFeatures(
    nFeatures: int,
    maxFeatures?: number | 'auto' | 'sqrt' | 'log2'
  ) {
    if (maxFeatures === 'log2') {
      return Math.floor(Math.log2(nFeatures))
    }
    if (maxFeatures === 'sqrt') {
      return Math.floor(Math.sqrt(nFeatures))
    }
    if (maxFeatures === 'auto') {
      return Math.floor(Math.sqrt(nFeatures))
    }
    if (typeof maxFeatures === 'number') {
      assert(maxFeatures >= 1, 'maxFeatures must be greater than 1')
      return Math.min(Math.floor(maxFeatures), nFeatures)
    }

    return nFeatures
  }
  public fit(X: number[][], y: int[], samplesSubset?: number[]) {
    this.X = X
    this.y = y

    let newSamplesSubset = samplesSubset || []

    // CheckNegativeLabels(yptr);
    this.maxFeaturesNumb = this.calcMaxFeatures(X[0].length, this.maxFeatures)

    this.splitter = new Splitter(
      X,
      y,
      this.minSamplesLeaf,
      this.criterion,
      this.maxFeaturesNumb,
      newSamplesSubset
    )

    // put root node on stack
    let rootNode: NodeRecord = {
      start: 0,
      end: this.splitter.sampleMap.length,
      depth: 0,
      impurity: 0,
      nSamples: this.splitter.sampleMap.length,
      parentId: -1,
      isLeft: false
    }
    this.stack.push(rootNode)

    let isRootNode = true

    while (this.stack.length !== 0) {
      // take next node from stack
      let currentRecord = this.stack.pop() as NodeRecord
      this.splitter.resetSampleRange(currentRecord.start, currentRecord.end)
      let currentSplit: Split = makeDefaultSplit()

      let isLeaf =
        !(currentRecord.depth < this.maxDepth) ||
        currentRecord.nSamples < this.minSamplesSplit ||
        currentRecord.nSamples < 2 * this.minSamplesLeaf

      // evaluate abort criterion
      if (isRootNode) {
        currentRecord.impurity = this.splitter.criterion.nodeImpurity()
        isRootNode = false
      }

      // or currentRecord.impurity <= 0.0;
      // split unless isLeaf
      if (!isLeaf) {
        currentSplit = this.splitter.splitNode()
        isLeaf =
          isLeaf ||
          !currentSplit.foundSplit ||
          currentRecord.impurity <= this.minImpurityDecrease
      }

      let currentNode: Node = {
        parentId: currentRecord.parentId,
        impurity: currentRecord.impurity,
        isLeaf: isLeaf,
        isLeft: currentRecord.isLeft,
        nSamples: currentRecord.nSamples,
        splitFeature: currentSplit.feature,
        threshold: currentSplit.threshold,
        value: this.splitter.criterion.nodeValue().slice(),
        leftChildId: -1,
        rightChildId: -1
      }

      this.tree.nodes.push(currentNode)
      let nodeId = this.tree.nodes.length - 1

      if (!isLeaf) {
        let rightRecord: NodeRecord = {
          start: currentSplit.pos,
          end: currentRecord.end,
          nSamples: currentRecord.end - currentSplit.pos,
          depth: currentRecord.depth + 1,
          parentId: nodeId,
          isLeft: false,
          impurity: currentSplit.impurityRight
        }

        this.stack.push(rightRecord)

        let leftRecord: NodeRecord = {
          start: currentRecord.start,
          end: currentSplit.pos,
          nSamples: currentSplit.pos - currentRecord.start,
          depth: currentRecord.depth + 1,
          parentId: nodeId,
          isLeft: true,
          impurity: currentSplit.impurityLeft
        }

        this.stack.push(leftRecord)
      }
    }
    this.tree.populateChildIds()
    this.tree.isBuilt = true
  }
}

interface DecisionTreeClassifierParams {
  criterion?: 'gini' | 'entropy'
  maxDepth?: int
  minSamplesSplit?: number
  minSamplesLeaf?: number
  maxFeatures?: number | 'auto' | 'sqrt' | 'log2'
  minImpurityDecrease?: number
}
export class DecisionTreeClassifier extends DecisionTreeBase {
  labelEncoder: LabelEncoder
  constructor({
    criterion = 'gini',
    maxDepth = undefined,
    minSamplesSplit = 2,
    minSamplesLeaf = 1,
    maxFeatures = undefined,
    minImpurityDecrease = 0.0
  }: DecisionTreeClassifierParams = {}) {
    assert(
      ['gini', 'entropy'].includes(criterion as string),
      'For classification must pass either the "gini" or "entropy" criterion'
    )
    super({
      criterion,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      maxFeatures,
      minImpurityDecrease
    })
    this.labelEncoder = new LabelEncoder()
  }
  public fit(X: Scikit2D, y: Scikit1D): DecisionTreeClassifier {
    assert(isScikit1D(y), 'y value is not a 1D container')
    assert(isScikit2D(X), 'X value is not a 2D container')
    let XArray = convertScikit2DToArray(X)
    let yArray = convertScikit1DToArray(y)
    assert(XArray.length === yArray.length, 'X and y must be the same size')
    validateX(XArray) // checks to make sure there are no NaN's etc
    validateY(yArray) // checks to make sure there are no NaN's etc
    let yArrayFixed = this.labelEncoder.fitTransform(yArray)
    super.fit(
      XArray as number[][],
      convertScikit1DToArray(yArrayFixed) as number[]
    )
    return this
  }

  public getNLeaves() {
    return this.tree.nodes.filter((el) => el.isLeaf).length
  }
  public predict(X: Scikit2D) {
    assert(isScikit2D(X), 'X value is not a 2D container')
    let XArray = convertScikit2DToArray(X)
    validateX(XArray)
    let yValues = this.tree.predictClassification(XArray as number[][])
    return this.labelEncoder.inverseTransform(yValues)
  }

  public predictProba(X: number[][]) {
    return this.tree.predictProba(X)
  }

  public score(X: number[][], y: number[]): number {
    const yPred = this.predict(X)
    return accuracyScore(y, yPred)
  }
}

interface DecisionTreeRegressorParams {
  criterion?: 'squared_error'
  maxDepth?: int
  minSamplesSplit?: number
  minSamplesLeaf?: number
  maxFeatures?: number | 'auto' | 'sqrt' | 'log2'
  minImpurityDecrease?: number
}
export class DecisionTreeRegressor extends DecisionTreeBase {
  constructor({
    criterion = 'squared_error',
    maxDepth = undefined,
    minSamplesSplit = 2,
    minSamplesLeaf = 1,
    maxFeatures = undefined,
    minImpurityDecrease = 0.0
  }: DecisionTreeRegressorParams = {}) {
    assert(
      ['squared_error'].includes(criterion as string),
      'Must pass the regression criterion of "squared_error"'
    )
    super({
      criterion,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      maxFeatures,
      minImpurityDecrease
    })
  }
  public fit(X: Scikit2D, y: Scikit1D): DecisionTreeRegressor {
    assert(isScikit1D(y), 'y value is not a 1D container')
    assert(isScikit2D(X), 'X value is not a 2D container')
    let XArray = convertScikit2DToArray(X)
    let yArray = convertScikit1DToArray(y)
    assert(XArray.length === yArray.length, 'X and y must be the same size')
    validateX(XArray)
    // TODO yValidation for regression (check that there are no NaN's etc)
    super.fit(XArray as number[][], yArray as number[])
    return this
  }
  public getNLeaves() {
    return this.tree.nodes.filter((el) => el.isLeaf).length
  }
  public predict(X: number[][]) {
    return this.tree.predictRegression(X)
  }

  public score(X: number[][], y: number[]): number {
    const yPred = this.predict(X)
    return r2Score(y, yPred)
  }
}
