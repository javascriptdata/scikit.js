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
import { DataType } from '@tensorflow/tfjs-core/dist/types'
import {
  abs,
  DataTypeMap,
  lessEqual,
  max,
  sub,
  tensor,
  Tensor,
  Tensor1D,
  tensor1d,
  Tensor2D,
  tensor2d,
  TensorLike
} from '@tensorflow/tfjs'
import { DataFrame, Series } from 'danfojs'
import {
  ArrayType1D,
  ArrayType2D,
  Scikit1D,
  Scikit2D,
  ScikitVecOrMatrix,
  TypedArray
} from './types'
import { inferShape, isTypedArray } from './types.utils'

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
  if (data instanceof Series) {
    // Do type inference if no dtype is passed, otherwise try to parse as that dtype
    return dtype
      ? (data.tensor.asType(dtype) as Tensor1D)
      : (data.tensor as Tensor1D)
  }
  if (data instanceof Tensor) {
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
  return dtype ? tensor1d(data, dtype) : tensor1d(data)
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
  if (data instanceof DataFrame) {
    return dtype
      ? (data.tensor.asType(dtype) as Tensor2D)
      : (data.tensor as Tensor2D)
  }
  if (data instanceof Tensor) {
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
    return dtype ? tensor2d(newData, shape, dtype) : tensor2d(newData, shape)
  }

  return dtype
    ? tensor2d(data as any, undefined, dtype)
    : tensor2d(data as any, undefined)
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
  data: TensorLike | Tensor | DataFrame | Series,
  shape?: number[],
  dtype?: keyof DataTypeMap
): Tensor {
  if (data instanceof DataFrame) {
    return data.tensor
  }
  if (data instanceof Series) {
    return data.tensor
  }
  if (data instanceof Tensor) {
    let newData = data
    if (shape) {
      newData = newData.reshape(shape)
    }
    if (dtype) {
      newData = newData.asType(dtype)
    }
    return newData
  }
  return tensor(data, shape, dtype)
}
export function convertTensorToInputType(
  tensor: Tensor,
  inputData: ScikitVecOrMatrix
) {
  if (inputData instanceof Tensor) {
    return tensor
  } else if (inputData instanceof DataFrame) {
    return new DataFrame(tensor, {
      index: inputData.index,
      columns: inputData.columns
    })
  } else if (inputData instanceof Series) {
    return new Series(tensor, {
      index: inputData.index
    })
  } else if (Array.isArray(inputData)) {
    return tensor.arraySync()
  } else {
    return tensor
  }
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
  return Boolean(lessEqual(max(abs(sub(tensor1, tensor2))), tol).dataSync()[0])
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

export function convertTo2DArray(data: Scikit2D) {
  if (data instanceof DataFrame) {
    return data.values
  }
  if (data instanceof Tensor) {
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
