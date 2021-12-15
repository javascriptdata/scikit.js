import {
  Tensor,
  Tensor1D,
  tensor1d,
  tensor2d,
  Tensor2D
} from '@tensorflow/tfjs'
import { Scikit1D, Scikit2D } from '../types'
import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
import { tf } from '../../globals'

export interface KNeighborsRegressorParams {
  nNeighbors?: number
  weights?: 'uniform' | 'distance'
  algorithm?: 'brute'
}

export class KNeighborsRegressor {
  nNeighbors: number
  weights: string
  algorithm: string
  X: Tensor2D
  y: Tensor
  constructor({
    nNeighbors = 5,
    weights = 'uniform',
    algorithm = 'brute'
  }: KNeighborsRegressorParams = {}) {
    this.nNeighbors = nNeighbors
    this.weights = weights
    this.algorithm = algorithm
    this.X = tensor2d([[]])
    this.y = tensor1d([])
  }

  getEuclideanDistances(X: Tensor2D, queryPoints: Tensor2D) {
    const expandedX = tf.expandDims(X, 0)
    const expandedClusters = tf.expandDims(queryPoints, 1)
    return tf.squaredDifference(expandedX, expandedClusters).sum(2).sqrt()
  }

  fit(X: Scikit2D, y: Scikit1D) {
    let X2D = convertToNumericTensor2D(X)
    let y1D = convertToNumericTensor1D(y)
    this.X = X2D
    this.y = y1D
  }

  predict(X: Scikit2D): Tensor1D {
    let query = convertToNumericTensor2D(X)
    let distancesMatrix = this.getEuclideanDistances(this.X, query)

    /**
     * Tensorflow doesn't have a smallestK function but they do have a topK which
     * finds the k biggest elements in a list. We are going to multiply our distances
     * by -1, use the "topK", then multiply by negative 1 to turn them back to our
     * smallestK
     */
    let closestPointsAndIndices = tf.topk(
      distancesMatrix.mul(-1),
      this.nNeighbors
    )
    let points = this.y.gather(closestPointsAndIndices.indices)
    let avg = points.div(this.nNeighbors).sum<Tensor1D>(1)
    return avg
  }
}
