import {
  div,
  equal,
  scalar,
  squaredDifference,
  sum,
  Tensor1D,
  tensor2d,
  Tensor2D,
  tidy
} from '@tensorflow/tfjs-core'
import { tf } from '../../globals'

/*
Next steps
1. Implement correct tol, maxIter logic
2. Implement correct nIter logic
3. Implement transform logic
4. Make it pass next 5 tests in sklearn test logic
*/

// Modified Fisher-Yates algorithm which takes
// a seed and selects n random numbers from a
// set of integers going from 0 to size-1
function sampleWithoutReplacement(size: number, n: number, seed?: number) {
  let curMap = new Map<number, number>()
  let finalNumbs = []
  let randoms = tf.randomUniform([n], 0, size, 'float32', seed).dataSync()
  for (let i = 0; i < randoms.length; i++) {
    randoms[i] = (randoms[i] * (size - i)) / size
    let randInt = Math.floor(randoms[i])
    let lastIndex = size - i - 1
    if (curMap.get(randInt) === undefined) {
      curMap.set(randInt, randInt)
    }
    if (curMap.get(lastIndex) === undefined) {
      curMap.set(lastIndex, lastIndex)
    }
    let holder = curMap.get(lastIndex) as number
    curMap.set(lastIndex, curMap.get(randInt) as number)
    curMap.set(randInt, holder)
    finalNumbs.push(curMap.get(lastIndex) as number)
  }

  return finalNumbs
}

export interface KMeansParams {
  nClusters?: number
  init?: 'kmeans++' | 'random'
  nInit?: number
  maxIter?: number
  tol?: number
  randomState?: number
}

/**
 * KMeans aims to cluster the input
 */
export default class KMeans {
  /* Constructor Args */
  nClusters: number
  init: string
  nInit?: number
  maxIter: number
  tol: number
  randomState?: number

  // Attributes
  clusterCenters: Tensor2D

  constructor({
    nClusters = 8,
    init = 'random',
    maxIter = 300,
    tol = 0.0001,
    nInit = 10,
    randomState
  }: KMeansParams = {}) {
    this.nClusters = nClusters
    this.init = init
    this.maxIter = maxIter
    this.tol = tol
    this.randomState = randomState
    this.nInit = nInit
    this.clusterCenters = tensor2d([[]])
  }

  initCentroids(X: Tensor2D) {
    if (this.init === 'random') {
      let indices = sampleWithoutReplacement(
        X.shape[0],
        this.nClusters,
        this.randomState
      )
      this.clusterCenters = tf.gather(X, indices)
      return
    }
    throw new Error(`init ${this.init} not implemented currently`)
  }

  closestCentroid(X: Tensor2D, strategy = 'euclidean'): Tensor1D {
    return tidy(() => {
      const expandedX = tf.expandDims(X, 1)
      const expandedClusters = tf.expandDims(this.clusterCenters, 0)
      return squaredDifference(expandedX, expandedClusters).sum(2).argMin(1)
    })
  }

  updateCentroids(X: Tensor2D, nearestIndices: Tensor1D): Tensor2D {
    return tidy(() => {
      const newCentroids = []
      for (let i = 0; i < this.nClusters; i++) {
        const mask = equal(nearestIndices, scalar(i).toInt())
        const currentCentroid = div(
          // set all masked instances to 0 by multiplying the mask tensor,
          // then sum across all instances
          sum(tf.mul(tf.expandDims(mask.toFloat(), 1), X), 0),
          // divided by number of instances
          sum(mask.toFloat())
        )
        newCentroids.push(currentCentroid)
      }
      return tf.stack(newCentroids) as Tensor2D
    })
  }
  fit(X: Tensor2D): KMeans {
    this.initCentroids(X)
    for (let i = 0; i < this.maxIter; i++) {
      const centroidPicks = this.closestCentroid(X)
      this.clusterCenters = this.updateCentroids(X, centroidPicks)
    }
    return this
  }
  predict(X: Tensor2D): Tensor1D {
    return this.closestCentroid(X)
  }
}
