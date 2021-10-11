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
import { TensorLike1D, TensorLike2D } from '@tensorflow/tfjs-core/dist/types'
import {
  DataTypeMap,
  tensor,
  Tensor,
  Tensor1D,
  tensor1d,
  Tensor2D,
  tensor2d,
  TensorLike,
} from '@tensorflow/tfjs-node'
import { DataFrame, Series } from 'danfojs-node'
import { ArrayType1D, ArrayType2D } from 'types'

export type Scikit1D = TensorLike1D | Tensor1D | Series
export type Scikit2D = TensorLike2D | Tensor2D | DataFrame
export type ScikitVecOrMatrix = Scikit1D | Scikit2D

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
export function convertToTensor1D(data: Scikit1D): Tensor1D {
  if (data instanceof Series) {
    return tensor1d(data.values)
  }
  if (data instanceof Tensor) {
    if (data.shape.length === 1) {
      return data
    } else {
      throw new Error(
        'ParamError: if data is a Tensor it must be a Tensor1D. If you really meant to reshape this tensor than use tf.reshape'
      )
    }
  }
  return tensor1d(data)
}

export function convertToTensor2D(data: Scikit2D): Tensor2D {
  if (data instanceof DataFrame) {
    return tensor2d(data.values)
  }
  if (data instanceof Tensor) {
    if (data.shape.length === 2) {
      return data
    } else {
      throw new Error(
        'ParamError: if data is a Tensor it must be a Tensor2D. If you really meant to reshape this tensor than use tf.reshape'
      )
    }
  }
  return tensor2d(data)
}

export function convertToTensor1D_2D(
  data: ScikitVecOrMatrix
): Tensor1D | Tensor2D {
  try {
    const new1DTensor = convertToTensor1D(data as Tensor1D)
    return new1DTensor
  } catch (e) {
    try {
      const new2DTensor = convertToTensor2D(data as Tensor2D)
      return new2DTensor
    } catch (newE) {
      throw new Error('ParamError: Can"t convert data into 1D or 2D tensor')
    }
  }
}

export function assertSameShape(a: Tensor, b: Tensor) {
  if (a.shape.length !== b.shape.length) {
    throw new Error(
      'ParamError: Shapes don"t match for these two Tensors. They are different dimensions'
    )
  }
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) {
      throw new Error('ParamError: Shapes do not match for these tensors')
    }
  }
}

export function assertSameType(a: Tensor, b: Tensor) {
  if (a.dtype !== b.dtype) {
    throw new Error('ParamError: These two Tensors do not have the same dtype')
  }
}

export function assertSameShapeAndType(a: Tensor, b: Tensor) {
  assertSameShape(a, b)
  assertSameType(a, b)
}

export function convertToTensor(
  data: TensorLike | Tensor | DataFrame | Series,
  shape?: number[],
  dtype?: keyof DataTypeMap
): Tensor {
  if (data instanceof DataFrame) {
    return tensor2d(data.values, shape && [shape[0], shape[1]], dtype)
  }
  if (data instanceof Series) {
    return tensor1d(data.values, dtype)
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
