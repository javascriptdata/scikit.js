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
  Scalar,
  Tensor1D,
  Tensor2D,
  TensorLike,
  Tensor,
  RecursiveArray,
  DataType
} from '@tensorflow/tfjs-core/dist/index.d'

import {
  ModelCompileArgs,
  ModelFitArgs
} from '@tensorflow/tfjs-layers/dist/index.d'
/////////////////////////Danfo types

export interface NDframeInterface {
  config?: any
  $setDtypes(dtypes: Array<string>, infer: boolean): void
  $setIndex(index: Array<string | number>): void
  $resetIndex(): void
  $setColumnNames(columns: string[]): void
  get dtypes(): Array<string>
  get ndim(): number
  get axis(): any
  get index(): Array<string | number>
  get columns(): string[]
  get shape(): Array<number>
  get values(): ArrayType1D | ArrayType2D
  get tensor(): any
  get size(): number
  print(): void
}
//End of Generic class types

export interface SeriesInterface extends NDframeInterface {
  iloc(rows: Array<string | number> | boolean[]): any
  head(rows: number): any
  tail(rows: number): any
  sample(num: number, options?: { seed?: number }): Promise<any>
  add(
    other: any | number | Array<number>,
    options?: { inplace?: boolean }
  ): any | void
  sub(
    other: any | number | Array<number>,
    options?: { inplace?: boolean }
  ): any | void
  mul(
    other: any | number | Array<number>,
    options?: { inplace?: boolean }
  ): any | void
  div(
    other: any | number | Array<number>,
    options?: { inplace?: boolean }
  ): any | void
  pow(
    other: any | number | Array<number>,
    options?: { inplace?: boolean }
  ): any | void
  mod(
    other: any | number | Array<number>,
    options?: { inplace?: boolean }
  ): any | void
  mean(): number
  median(): number
  mode(): any
  min(): number
  max(): number
  sum(): number
  count(): number
  maximum(other: any | number | Array<number>): any
  minimum(other: any | number | Array<number>): any
  round(dp: number, options?: { inplace?: boolean }): any | void
  std(): number
  var(): number
  isNa(): any
  fillNa(
    value: number | string | boolean,
    options?: { inplace?: boolean }
  ): any | void
  sortValues(options?: { ascending?: boolean; inplace?: boolean }): any | void
  copy(): any
  describe(): any
  resetIndex(options?: { inplace?: boolean }): any | void
  setIndex(
    index: Array<number | string | (number | string)>,
    options?: { inplace?: boolean }
  ): any | void
  map(callable: any, options?: { inplace?: boolean }): any | void
  apply(
    callable: (value: any) => any,
    options?: { inplace?: boolean }
  ): any | void
  unique(): any
  nUnique(): number
  valueCounts(): any
  abs(options?: { inplace?: boolean }): any | void
  cumSum(options?: { inplace?: boolean }): any | void
  cumMin(options?: { inplace?: boolean }): any | void
  cumMax(options?: { inplace?: boolean }): any | void
  cumProd(options?: { inplace?: boolean }): any | void
  lt(other: any | number | Array<number> | boolean[]): any
  gt(other: any | number | Array<number> | boolean[]): any
  le(other: any | number | Array<number> | boolean[]): any
  ge(other: any | number | Array<number> | boolean[]): any
  ne(other: any | number | Array<number> | boolean[]): any
  eq(other: any | number | Array<number> | boolean[]): any
  replace(
    oldValue: string | number | boolean,
    newValue: string | number | boolean,
    options?: { inplace?: boolean }
  ): any | void
  dropNa(options?: { inplace?: boolean }): any | void
  argSort(options?: { ascending: boolean }): any
  argMax(): number
  argMin(): number
  get dtype(): string
  dropDuplicates(options?: {
    keep?: 'first' | 'last'
    inplace?: boolean
  }): any | void
  asType(
    dtype: 'float32' | 'int32' | 'string' | 'boolean',
    options?: { inplace?: boolean }
  ): any | void
  get str(): any
  get dt(): any
  append(
    values: string | number | boolean | any | ArrayType1D,
    index: Array<number | string> | number | string,
    options?: { inplace?: boolean }
  ): any | void
  toString(): string
  and(other: any): any
  or(other: any): any
  getDummies(options?: {
    prefix?: string | Array<string>
    prefixSeparator?: string | Array<string>
    inplace?: boolean
  }): any
  iat(index: number): number | string | boolean | undefined
  at(index: string | number): number | string | boolean | undefined
  plot(divId: string): any
}

//Start of DataFrame class types
export interface DataFrameInterface extends NDframeInterface {
  [key: string]: any
  drop(options: {
    columns?: string | Array<string>
    index?: Array<string | number>
    inplace?: boolean
  }): any | void
  loc(options: { rows?: Array<string | number>; columns?: Array<string> }): any
  iloc(options: {
    rows?: Array<string | number>
    columns?: Array<string | number>
  }): any
  head(rows?: number): any
  tail(rows?: number): any
  sample(num: number, options?: { seed?: number }): Promise<any>
  add(
    other: any | any | number | number[],
    options?: { axis?: 0 | 1; inplace?: boolean }
  ): any | void
  sub(
    other: any | any | number | number[],
    options?: { axis?: 0 | 1; inplace?: boolean }
  ): any | void
  mul(
    other: any | any | number | number[],
    options?: { axis?: 0 | 1; inplace?: boolean }
  ): any | void
  div(
    other: any | any | number | number[],
    options?: { axis?: 0 | 1; inplace?: boolean }
  ): any | void
  pow(
    other: any | any | number | number[],
    options?: { axis?: 0 | 1; inplace?: boolean }
  ): any | void
  mod(
    other: any | any | number | number[],
    options?: { axis?: 0 | 1; inplace?: boolean }
  ): any | void
  mean(options?: { axis?: 0 | 1 }): any
  median(options?: { axis?: 0 | 1 }): any
  mode(options?: { axis?: 0 | 1; keep?: number }): any
  min(options?: { axis?: 0 | 1 }): any
  max(options?: { axis?: 0 | 1 }): any
  std(options?: { axis?: 0 | 1 }): any
  var(options?: { axis?: 0 | 1 }): any
  sum(options?: { axis?: 0 | 1 }): any
  count(options?: { axis?: 0 | 1 }): any
  round(dp?: number, options?: { inplace: boolean }): any | void
  cumSum(options?: { axis?: 0 | 1 }): any | void
  cumMin(options?: { axis?: 0 | 1 }): any | void
  cumMax(options?: { axis?: 0 | 1 }): any | void
  cumProd(options?: { axis?: 0 | 1 }): any | void
  copy(): any
  resetIndex(options: { inplace?: boolean }): any | void
  setIndex(options: {
    index: Array<number | string | (number | string)>
    column?: string
    drop?: boolean
    inplace?: boolean
  }): any | void
  describe(): any
  selectDtypes(include: Array<string>): any
  abs(options?: { inplace?: boolean }): any | void
  query(
    condition: any | Array<boolean>,
    options?: { inplace?: boolean }
  ): any | void
  addColumn(
    column: string,
    values: any | ArrayType1D,
    options?: {
      inplace?: boolean
      atIndex?: number | string
    }
  ): any | void
  groupby(col: Array<string>): any
  column(column: string): any
  fillNa(
    value: ArrayType1D,
    options?: {
      columns?: Array<string>
      inplace?: boolean
    }
  ): any | void
  isNa(): any
  dropNa(options?: { axis: 0 | 1; inplace?: boolean }): any | void
  apply(callable: any, options?: { axis?: 0 | 1 }): any | any
  applyMap(callable: any, options?: { inplace?: boolean }): any | void
  lt(other: any | any | number, options?: { axis?: 0 | 1 }): any
  gt(other: any | any | number, options?: { axis?: 0 | 1 }): any
  le(other: any | any | number, options?: { axis?: 0 | 1 }): any
  ge(other: any | any | number, options?: { axis?: 0 | 1 }): any
  ne(other: any | any | number, options?: { axis?: 0 | 1 }): any
  eq(other: any | any | number, options?: { axis?: 0 | 1 }): any
  replace(
    oldValue: number | string | boolean,
    newValue: number | string | boolean,
    options?: {
      columns?: Array<string>
      inplace?: boolean
    }
  ): any | void
  transpose(options?: { inplace?: boolean }): any | void
  get T(): any
  get ctypes(): any
  asType(
    column: string,
    dtype: 'float32' | 'int32' | 'string' | 'boolean',
    options?: { inplace?: boolean }
  ): any | void
  nUnique(axis: 0 | 1): any
  rename(
    mapper: object,
    options?: {
      axis?: 0 | 1
      inplace?: boolean
    }
  ): any | void
  sortIndex(options?: { inplace?: boolean; ascending?: boolean }): any | void
  sortValues(
    column: string,
    options?: {
      inplace?: boolean
      ascending?: boolean
    }
  ): any | void
  append(
    newValues: ArrayType1D | ArrayType2D | any | any,
    index: Array<number | string> | number | string,
    options?: {
      inplace?: boolean
    }
  ): any | void
  toString(): string
  getDummies(options?: {
    columns?: string | Array<string>
    prefix?: string | Array<string>
    prefixSeparator?: string | Array<string>
    inplace?: boolean
  }): any | void
  iat(row: number, column: number): number | string | boolean | undefined
  at(
    row: string | number,
    column: string
  ): number | string | boolean | undefined
  plot(divId: string): any
}

////////////////////////////////////
// The Types that Scikit uses
export {
  Tensor1D,
  Tensor2D,
  TensorLike,
  Tensor,
  ModelCompileArgs,
  ModelFitArgs,
  RecursiveArray,
  Scalar,
  DataType
}
export type TypedArray = Float32Array | Int32Array | Uint8Array
export type ScikitLike1D = TypedArray | number[] | boolean[] | string[]
export type ScikitLike2D = TypedArray[] | number[][] | boolean[][] | string[][]
export type Scikit1D = ScikitLike1D | Tensor1D | SeriesInterface
export type Scikit2D = ScikitLike2D | Tensor2D | DataFrameInterface
export type ScikitVecOrMatrix = Scikit1D | Scikit2D
export type OptimizerTypes =
  | 'sgd'
  | 'momentum'
  | 'adadelta'
  | 'rmsprop'
  | 'adamax'
  | 'adam'
  | 'adagrad'
export type LossTypes =
  | 'meanSquaredError'
  | 'sigmoidCrossEntropy'
  | 'softmaxCrossEntropy'
  | 'logLoss'
  | 'huberLoss'
  | 'hingeLoss'
  | 'cosineDistance'
  | 'computeWeightedLoss'
  | 'absoluteDifference'
  | 'custom'

export type Initializers = 'Zeros' | 'Ones'
export type ArrayType1D = Array<
  number | string | boolean | (number | string | boolean)
>

export type ArrayType2D = Array<
  number[] | string[] | boolean[] | (number | string | boolean)[]
>

export type Iterable<K> = {
  [index: number]: K
  length: number
}

export type int = number

export interface Transformer {
  fit(X: Scikit2D, y?: Scikit1D): any
  transform(X: Scikit2D, y?: Scikit1D): Tensor2D
  fitTransform(X: Scikit2D, y?: Scikit1D): Tensor2D
}
