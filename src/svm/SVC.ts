// import { tf } from '../shared/globals'
// import { Scikit1D, Scikit2D } from '../index'
// import { SVM, SVMParam, KERNEL_TYPE, ISVMParam, SVM_TYPE } from 'libsvm-wasm'
// import { convertToNumericTensor1D, convertToNumericTensor2D } from '../utils'
// import { assert } from '../typesUtils'

// export interface SVCParams {
//   kernel?: 'LINEAR' | 'POLY' | 'RBF' | 'SIGMOID' | 'PRECOMPUTED'
//   degree?: number
//   gamma?: number | 'auto' | 'scale'
//   coef0?: number
//   tol?: number
//   C?: number
//   epsilon?: number
//   shrinking?: boolean
//   cacheSize?: number
//   maxIter?: number
//   classWeight?: { [key: number]: number } | 'balanced'
// }

// export class SVC {
//   private svm?: SVM
//   private svmParam: SVMParam
//   private gammaMode = 'scale'
//   private classWeight: { [key: number]: number } | 'balanced' | undefined

//   constructor({
//     kernel = 'RBF',
//     degree = 3,
//     gamma = 'scale',
//     coef0 = 0,
//     tol = 1e-3,
//     C = 1,
//     epsilon = 0.1,
//     shrinking = true,
//     cacheSize = 200,
//     classWeight = {},
//     maxIter = -1
//   }: SVCParams = {}) {
//     const inernalSVMParam: ISVMParam = {
//       kernel_type: KERNEL_TYPE[kernel],
//       svm_type: SVM_TYPE.C_SVC,
//       degree,
//       coef0,
//       C,
//       p: epsilon,
//       shrinking: shrinking ? 1 : 0,
//       cache_size: cacheSize
//     }
//     if (gamma === 'auto') {
//       this.gammaMode = gamma
//     } else if (gamma === 'scale') {
//       this.gammaMode = gamma
//     } else {
//       inernalSVMParam.gamma = gamma
//     }

//     this.classWeight = undefined

//     this.svmParam = new SVMParam(inernalSVMParam, tol)
//   }

//   async fit(X: Scikit2D, y: Scikit1D): Promise<SVC> {
//     let XTwoD = convertToNumericTensor2D(X)
//     let yOneD = convertToNumericTensor1D(y)

//     let nSample = XTwoD.shape[0]
//     let nFeature = XTwoD.shape[1]

//     assert(
//       yOneD.shape[0] === nSample,
//       'X and y must have the same number of samples'
//     )
//     assert(yOneD.shape[0] >= 1, 'Must have more than 1 sample in X, and y')
//     // Sum((XTwoD - Mean) ** 2) / nSample
//     const VarianceOfX = tf
//       .squaredDifference(XTwoD, XTwoD.mean())
//       .sum()
//       .div(nSample)
//       .dataSync()[0]

//     if (this.gammaMode === 'scale') {
//       this.svmParam.param.gamma = 1 / (nFeature * VarianceOfX)
//     } else if (this.gammaMode === 'auto') {
//       this.svmParam.param.gamma = 1 / nFeature
//     }

//     const [processX, processY] = await Promise.all([
//       XTwoD.array(),
//       yOneD.array()
//     ])

//     const labelSet = new Set(processY)
//     const numLabels = labelSet.size
//     const weightLabel = new Array(numLabels)
//     const weight = new Array(numLabels)
//     let idx = 0

//     if (this.classWeight) {
//       for (let label of labelSet) {
//         if (this.classWeight[label] === undefined) {
//           throw new Error('Class weight not found')
//         } else {
//           weightLabel[idx] = label
//           weight[idx] = this.classWeight[label]
//           idx++
//         }
//       }
//     } else {
//       for (let label of labelSet) {
//         weightLabel[idx] = label
//         weight[idx] = 1
//         idx++
//       }
//     }

//     this.svmParam.param.weight_label = weightLabel
//     this.svmParam.param.weight = weight
//     this.svmParam.param.nr_weight = numLabels

//     this.svm = new SVM(this.svmParam)
//     await this.svm.init()
//     await this.svm.feedSamples(processX, processY)
//     await this.svm.train()
//     return this
//   }

//   predict(X: Scikit2D): tf.Tensor1D {
//     const XTensor = convertToNumericTensor2D(X)
//     const processX = XTensor.arraySync()
//     assert(Boolean(this.svm), 'SVM was not trained')

//     const results = processX.map((el) => (this.svm as any).predict(el))
//     return tf.tensor1d(results)
//   }
// }
