/**
*  @license
* Copyright 2021, JsData. All rights reserved.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==========================================================================
*/

import {
  ArrayType1D,
  ArrayType2D,
  DataFrameInterface,
  Initializers,
  LossTypes,
  OptimizerTypes,
  Scikit1D,
  Scikit2D,
  ScikitVecOrMatrix,
  SeriesInterface,
  TypedArray,
  DataType,
  Tensor2D,
  Tensor1D,
  Tensor,
  TensorLike
} from './types'
import {
  assert,
  inferShape,
  isDataFrameInterface,
  isScikitVecOrMatrix,
  isSeriesInterface,
  isTypedArray,
  isTensor
} from './typesUtils'
import { getBackend } from './tf-singleton'

/**
 * Generates an array of dim (row x column) with inner values set to zero
 * @param row
 * @param column
 */
export const zeros = (
  row: number,
  column: number
): ArrayType1D | ArrayType2D => {
  const zeroData = []
  for (let i = 0; i < row; i++) {
    const colData = Array(column)
    for (let j = 0; j < column; j++) {
      colData[j] = 0
    }
    zeroData.push(colData)
  }
  return zeroData
}

/**
 * Checks if array is 1D
 * @param arr The array
 */
export const is1DArray = (arr: ArrayType1D | ArrayType2D): boolean => {
  if (
    typeof arr[0] == 'number' ||
    typeof arr[0] == 'string' ||
    typeof arr[0] == 'boolean'
  ) {
    return true
  } else {
    return false
  }
}
/**
 *
 * @param data Scikit1D One dimensional array of data
 * @returns Tensor1D. If you pass in something that isn't 1D, then it will throw an error.
 * This is the case with 2D Tensors as well. If you really want to reshape them then use tf.reshape
 */
export function convertToTensor1D(data: Scikit1D, dtype?: DataType): Tensor1D {
  let tf = getBackend()
  if (isSeriesInterface(data)) {
    // Do type inference if no dtype is passed, otherwise try to parse as that dtype
    return dtype
      ? (data.tensor.asType(dtype) as unknown as Tensor1D)
      : (data.tensor as unknown as Tensor1D)
  }
  if (isTensor(data)) {
    if (data.shape.length === 1) {
      if (!dtype || data.dtype == dtype) {
        return data
      }
      return data.asType(dtype)
    } else {
      throw new Error(
        'ParamError: if data is a Tensor it must be a Tensor1D. If you really meant to reshape this tensor than use tf.reshape'
      )
    }
  }
  return dtype ? tf.tensor1d(data, dtype) : tf.tensor1d(data)
}

export function convertToNumericTensor1D(data: Scikit1D, dtype?: DataType) {
  const newTensor = convertToTensor1D(data, dtype)
  if (newTensor.dtype === 'string') {
    throw new Error(
      "ParamError: data has string dtype, can't convert to numeric Tensor"
    )
  }
  return newTensor
}

export function convertToTensor2D(data: Scikit2D, dtype?: DataType): Tensor2D {
  let tf = getBackend()
  if (isDataFrameInterface(data)) {
    return dtype
      ? (data.tensor.asType(dtype) as unknown as Tensor2D)
      : (data.tensor as unknown as Tensor2D)
  }
  if (isTensor(data)) {
    if (data.shape.length === 2) {
      if (!dtype || data.dtype == dtype) {
        return data
      }
      return data.asType(dtype)
    } else {
      throw new Error(
        'ParamError: if data is a Tensor it must be a Tensor2D. If you really meant to reshape this tensor than use tf.reshape'
      )
    }
  }
  if (Array.isArray(data) && isTypedArray(data[0])) {
    const shape = inferShape(data) as [number, number]
    const newData = data.map((el) => Array.from(el as number[]))
    return dtype
      ? tf.tensor2d(newData, shape, dtype)
      : tf.tensor2d(newData, shape)
  }

  return dtype
    ? tf.tensor2d(data as any, undefined, dtype)
    : tf.tensor2d(data as any, undefined)
}

export function convertToTensor1D_2D(
  data: ScikitVecOrMatrix,
  dtype?: DataType
): Tensor1D | Tensor2D {
  try {
    const new1DTensor = convertToTensor1D(data as Tensor1D, dtype)
    return new1DTensor
  } catch (e) {
    try {
      const new2DTensor = convertToTensor2D(data as Tensor2D, dtype)
      return new2DTensor
    } catch (newE) {
      throw new Error('ParamError: Can"t convert data into 1D or 2D tensor')
    }
  }
}

export function convertToNumericTensor2D(data: Scikit2D, dtype?: DataType) {
  const newTensor = convertToTensor2D(data, dtype)
  if (newTensor.dtype === 'string') {
    throw new Error(
      "ParamError: data has string dtype, can't convert to numeric Tensor"
    )
  }
  return newTensor
}

export function convertToNumericTensor1D_2D(
  data: ScikitVecOrMatrix,
  dtype?: DataType
) {
  const newTensor = convertToTensor1D_2D(data, dtype)
  if (newTensor.dtype === 'string') {
    throw new Error(
      "ParamError: data has string dtype, can't convert to numeric Tensor"
    )
  }
  return newTensor
}

export function convertToTensor(
  data: TensorLike | Tensor | DataFrameInterface | SeriesInterface,
  shape?: number[],
  dtype?: DataType
): Tensor {
  let tf = getBackend()
  if (isDataFrameInterface(data)) {
    return data.tensor as unknown as Tensor2D
  }
  if (isSeriesInterface(data)) {
    return data.tensor as unknown as Tensor2D
  }
  if (isTensor(data)) {
    let newData = data
    if (shape) {
      newData = newData.reshape(shape)
    }
    if (dtype) {
      newData = newData.asType(dtype)
    }
    return newData
  }
  return tf.tensor(data, shape, dtype)
}

/**
 * Check that if two tensor are of same shape
 * @param tensor1
 * @param tensor2
 * @returns
 */
export const shapeEqual = (tensor1: Tensor, tensor2: Tensor): boolean => {
  const shape1 = tensor1.shape
  const shape2 = tensor2.shape
  if (shape1.length != shape2.length) {
    return false
  }
  for (let i = 0; i < shape1.length; i++) {
    if (shape1[i] !== shape2[i]) {
      return false
    }
  }
  return true
}

/**
 * Check that two tensors are equal to within some additive tolerance.
 * @param tensor1
 * @param tensor2
 * @param
 */
export const tensorEqual = (
  tensor1: Tensor,
  tensor2: Tensor,
  tol = 0
): boolean => {
  if (!shapeEqual(tensor1, tensor2)) {
    throw new Error('tensor1 and tensor2 not of same shape')
  }
  let tf = getBackend()
  return Boolean(
    tf.lessEqual(tf.max(tf.abs(tf.sub(tensor1, tensor2))), tol).dataSync()[0]
  )
}

export const arrayEqual = (
  array: Array<any> | any,
  array2: Array<any> | any,
  tol = 0
): boolean => {
  if (!Array.isArray(array) && !Array.isArray(array2)) {
    return Math.abs(array - array2) <= tol
  }
  if (array.length !== array2.length) {
    return false
  }

  for (let i = 0; i < array.length; i++) {
    if (!arrayEqual(array[i], array2[i], tol)) {
      return false
    }
  }
  return true
}

export function convertScikit2DToArray(
  data: Scikit2D
): any[][] | TypedArray[] {
  if (isDataFrameInterface(data)) {
    return data.values as any[][]
  }
  if (isTensor(data)) {
    return data.arraySync()
  }
  return data
}

export function convertScikit1DToArray(data: Scikit1D): any[] | TypedArray {
  if (isSeriesInterface(data)) {
    return data.values
  }
  if (isTensor(data)) {
    return data.arraySync()
  }
  return data
}

export function arrayTo2DColumn(array: any[] | TypedArray) {
  let newArray = []
  for (let i = 0; i < array.length; i++) {
    newArray.push([array[i]])
  }
  return newArray
}

export function getLength(X: Scikit2D | Scikit1D): number {
  assert(isScikitVecOrMatrix(X), "X isn't a Scikit2D or Scikit1D object")
  if (isTensor(X)) {
    return X.shape[0]
  }
  if (isDataFrameInterface(X) || isSeriesInterface(X)) {
    return X.size
  }
  return X.length
}

/**
 * Modified Fisher-Yates algorithm which takes
 * a seed and selects n random numbers from a
 * set of integers going from 0 to size-1
 */
export function sampleWithoutReplacement(
  size: number,
  n: number,
  seed?: number
) {
  let tf = getBackend()
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

export function optimizer(opt: OptimizerTypes) {
  let tf = getBackend()
  switch (opt) {
    case 'sgd':
      return tf.train.sgd(0.1)
    case 'momentum':
      return tf.train.momentum(0.1, 0.9)
    case 'adadelta':
      return tf.train.adadelta()
    case 'adagrad':
      return tf.train.adagrad(0.1)
    case 'rmsprop':
      return tf.train.rmsprop(0.1)
    case 'adamax':
      return tf.train.adamax()
    case 'adam':
      return tf.train.adam()
  }
}

export function getLoss(loss: LossTypes) {
  let tf = getBackend()
  switch (loss) {
    case 'meanSquaredError':
      return tf.losses.meanSquaredError
    case 'sigmoidCrossEntropy':
      return tf.losses.sigmoidCrossEntropy
    case 'softmaxCrossEntropy':
      return tf.losses.softmaxCrossEntropy
    case 'logLoss':
      return tf.losses.logLoss
    case 'huberLoss':
      return tf.losses.huberLoss
    case 'hingeLoss':
      return tf.losses.hingeLoss
    case 'cosineDistance':
      return tf.losses.cosineDistance
    case 'computeWeightedLoss':
      return tf.losses.computeWeightedLoss
    case 'absoluteDifference':
      return tf.losses.absoluteDifference
    default:
      throw new Error(`${loss} loss not supported`)
  }
}

export function initializer(init: Initializers) {
  let tf = getBackend()
  switch (init) {
    case 'Zeros':
      return tf.initializers.zeros()
    case 'Ones':
      return tf.initializers.ones()
    default:
      throw new Error(`${init} initializer not supported`)
      break
  }
}
