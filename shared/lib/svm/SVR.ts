import { tf } from '../../globals'
import { Scikit1D, Scikit2D } from '../index'
import { SVM, SVMParam, KERNEL_TYPE, ISVMParam, SVM_TYPE } from 'libsvm-wasm'
import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
import { assert } from '../typesUtils'

export interface SVRParams {
  kernel?: 'LINEAR' | 'POLY' | 'RBF' | 'SIGMOID' | 'PRECOMPUTED'
  degree?: number
  gamma?: number | 'auto' | 'scale'
  coef0?: number
  tol?: number
  C?: number
  epsilon?: number
  shrinking?: boolean
  cacheSize?: number
  maxIter?: number
}

export class SVR {
  private svm?: SVM
  private svmParam: SVMParam
  private gammaMode: string = 'scale'

  constructor({
    kernel = 'RBF',
    degree = 3,
    gamma = 'scale',
    coef0 = 0,
    tol = 1e-3,
    C = 1,
    epsilon = 0.1,
    shrinking = true,
    cacheSize = 200,
    maxIter = -1
  }: SVRParams = {}) {
    const internalSVMParam: ISVMParam = {
      kernel_type: KERNEL_TYPE[kernel],
      svm_type: SVM_TYPE.EPSILON_SVR,
      degree,
      coef0,
      C,
      p: epsilon,
      shrinking: shrinking ? 1 : 0,
      cache_size: cacheSize
    }
    if (gamma === 'auto') {
      this.gammaMode = gamma
    } else if (gamma === 'scale') {
      this.gammaMode = gamma
    } else {
      internalSVMParam.gamma = gamma
    }

    this.svmParam = new SVMParam(internalSVMParam, tol)
  }

  async fit(X: Scikit2D, y: Scikit1D): Promise<SVR> {
    let XTwoD = convertToNumericTensor2D(X)
    let yOneD = convertToNumericTensor1D(y)
    let nSample = XTwoD.shape[0]
    let nFeature = XTwoD.shape[1]
    assert(
      yOneD.shape[0] === nSample,
      'X and y must have the same number of samples'
    )
    assert(yOneD.shape[0] >= 1, 'Must have more than 1 sample in X, and y')

    // Sum((XTwoD - Mean) ** 2) / nSample
    const VarianceOfX = tf
      .squaredDifference(XTwoD, XTwoD.mean())
      .sum()
      .div(nSample)
      .dataSync()[0]

    if (this.gammaMode === 'scale') {
      this.svmParam.param.gamma = 1 / (nFeature * VarianceOfX)
    } else if (this.gammaMode === 'auto') {
      this.svmParam.param.gamma = 1 / nFeature
    }

    const [processX, processY] = await Promise.all([
      XTwoD.array(),
      yOneD.array()
    ])

    this.svm = new SVM(this.svmParam)
    await this.svm.feedSamples(processX, processY)
    await this.svm.train()
    return this
  }

  async predict(X: Scikit2D): Promise<tf.Tensor1D> {
    const XTensor = convertToNumericTensor2D(X)
    const processX = await XTensor.array()
    assert(Boolean(this.svm), 'SVM was not trained')

    const results = processX.map((el) => (this.svm as any).predict(el))
    return tf.tensor1d(await Promise.all(results))
  }
}
