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

import { Tensor1D, Tensor2D } from '@tensorflow/tfjs-core'
import { dfd } from '../globals'

// The Types that Scikit uses
export type TypedArray = Float32Array | Int32Array | Uint8Array
export type ScikitLike1D = TypedArray | number[] | boolean[] | string[]
export type ScikitLike2D = TypedArray[] | number[][] | boolean[][] | string[][]
export type Scikit1D = ScikitLike1D | Tensor1D | dfd.Series
export type Scikit2D = ScikitLike2D | Tensor2D | dfd.DataFrame
export type ScikitVecOrMatrix = Scikit1D | Scikit2D

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

export type Strategy = 'mean' | 'median' | 'mostFrequent' | 'constant'

export interface Transformer {
  fit(X: Scikit2D, y?: Scikit1D): any
  transform(X: Scikit2D, y?: Scikit1D): Tensor2D
  fitTransform(X: Scikit2D, y?: Scikit1D): Tensor2D
}
