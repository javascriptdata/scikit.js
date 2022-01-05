/**
*  @license
* Copyright 2022, JsData. All rights reserved.
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

import { tf } from './shared/globals'
import { Tensor, TensorLike } from '@tensorflow/tfjs-core'

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace jest {
    interface Matchers<R, T> {
      /**
       * Tests whether or not each entry of a Tensor(Like) is close
       * to the corresponding entry of an expected tensor.
       *
       * @param expected The expected result tensor.
       * @param params Tolerance and broadcast settings. `{broadcast: false}` disallows
       *               broadcasting, i.e. the result tensor must have the same shape as
       *               the expected tensor. `{rtol, atol}` are the relative and absolute
       *               tolerance parameter. Two entries `x` and `y` are considered equal
       *               iff `|x-y| <= max(|x|, |y|)*rtol + atol`. Set `{rtol: 0, atol: 0}`
       *               for exact equality.
       */
      toBeAllCloseTo: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: { rtol?: number; atol?: number; broadcast?: boolean }
          ) => R
        : undefined
      toBeAllLessOrClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: { rtol?: number; atol?: number; broadcast?: boolean }
          ) => R
        : undefined
      toBeAllGreaterOrClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: { rtol?: number; atol?: number; broadcast?: boolean }
          ) => R
        : undefined
      toBeAllLessNotClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: { rtol?: number; atol?: number; broadcast?: boolean }
          ) => R
        : undefined
      toBeAllGreaterNotClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: { rtol?: number; atol?: number; broadcast?: boolean }
          ) => R
        : undefined
    }
  }
}

/**
 * Takes a flat index and turns it into an nd-index for
 * the given shape.
 *
 * @param flat Index into the flattened data array of a Tensor.
 * @param shape Shape of a Tensor.
 * @returns The nd-index into a Tensor of shape `shape`.
 */
function unravelIndex(flat: number, shape: number[]) {
  // see: https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
  shape = shape.slice()
  for (let i = shape.length; i-- > 0; ) {
    const si = shape[i]
    shape[i] = flat % si
    flat = Math.trunc(flat / si)
  }
  return shape
}

const isClose =
  ({ rtol = 1e-3, atol = 1e-6 }) =>
  (x: number, y: number) => {
    const xa = Math.abs(x)
    const ya = Math.abs(y)
    const tol = Math.max(xa, ya) * rtol + atol
    return Math.abs(x - y) <= tol
  }

const isLessOrClose =
  ({ rtol = 1e-3, atol = 1e-6 }) =>
  (x: number, y: number) => {
    const xa = Math.abs(x)
    const ya = Math.abs(y)
    const tol = Math.max(xa, ya) * rtol + atol
    return x - y <= tol
  }

const isLessNotClose =
  ({ rtol = 1e-3, atol = 1e-6 }) =>
  (x: number, y: number) => {
    const xa = Math.abs(x)
    const ya = Math.abs(y)
    const tol = Math.max(xa, ya) * rtol + atol
    return x - y < -tol
  }

export function toBeAll(
  this: { isNot: boolean },
  result: TensorLike | Tensor,
  expect: TensorLike | Tensor,
  { broadcast = true },
  description: string,
  match: (x: number, y: number) => boolean
) {
  const { isNot } = this
  const a = result instanceof Tensor ? result : tf.tensor(result)
  const b = expect instanceof Tensor ? expect : tf.tensor(expect)

  const msg = (msg: string) => () =>
    `\nA: ${a.toString(true)}\nB: ${b.toString(true)}\nExpected A ${
      isNot ? 'not ' : ''
    }to be all ${description} B but:\n${msg}`

  const rankA = a.rank
  const rankB = b.rank
  const rank = Math.max(rankA, rankB)

  const shapeA = a.shape
  const shapeB = b.shape
  const shape = [...(rankB < rankA ? shapeA : shapeB)]

  // CHECK SHAPES
  // ------------
  {
    let i = rankA
    let j = rankB

    if (broadcast) {
      while (i-- > 0 && j-- > 0) {
        const sa = shapeA[i]
        const sb = shapeB[j]
        if (sa !== sb && sa !== 1 && sb !== 1) {
          return {
            message: msg('A.shape not broadcast-compatible to B.shape'),
            pass: isNot
          }
        }
        shape[Math.max(i, j)] = Math.max(sa, sb)
      }
    } else {
      if (i !== j) {
        return {
          message: msg('A.shape does not match B.shape'),
          pass: isNot
        }
      }
      while (i-- > 0 && j-- > 0) {
        if (shapeA[i] !== shapeB[j]) {
          return {
            message: msg('A.shape does not match B.shape'),
            pass: isNot
          }
        }
      }
    }
  }

  // CHECK DATA
  // ----------
  const aFlat = a.dataSync()
  const bFlat = b.dataSync()

  let ia = 0
  let ib = 0
  let strideA: number
  let strideB: number

  function visit(axis: number) {
    if (axis === rank) {
      if (!match(aFlat[ia], bFlat[ib])) {
        throw msg(
          `A(${unravelIndex(ia, shapeA)}) = ${aFlat[ia]}\nB(${unravelIndex(
            ib,
            shapeB
          )}) = ${bFlat[ib]}`
        )
      }
      ia += strideA = 1
      ib += strideB = 1
    } else {
      for (let i = 0; ; ) {
        visit(axis + 1)
        if (++i >= shape[axis]) break
        if (!(shapeA[axis - rank + rankA] > 1)) ia -= strideA
        if (!(shapeB[axis - rank + rankB] > 1)) ib -= strideB
      }
      strideA *= shapeA[axis - rank + rankA] ?? 1
      strideB *= shapeB[axis - rank + rankB] ?? 1
    }
  }

  try {
    visit(0)
    return { pass: true, message: msg('It is.') }
  } catch (message) {
    return { pass: false, message: message as () => string }
  }
}

export function toBeAllCloseTo(
  this: { isNot: boolean },
  result: TensorLike | Tensor,
  expect: TensorLike | Tensor,
  params: { rtol?: number; atol?: number; broadcast?: boolean } = {}
) {
  return toBeAll.call(
    this,
    result,
    expect,
    params,
    'close to',
    isClose(params)
  )
}

export function toBeAllLessOrClose(
  this: { isNot: boolean },
  result: TensorLike | Tensor,
  expect: TensorLike | Tensor,
  params: { rtol?: number; atol?: number; broadcast?: boolean } = {}
) {
  return toBeAll.call(
    this,
    result,
    expect,
    params,
    'close to or less than',
    isLessOrClose(params)
  )
}

export function toBeAllGreaterOrClose(
  this: { isNot: boolean },
  result: TensorLike | Tensor,
  expect: TensorLike | Tensor,
  params: { rtol?: number; atol?: number; broadcast?: boolean } = {}
) {
  const le = isLessOrClose(params)
  return toBeAll.call(
    this,
    result,
    expect,
    params,
    'close to or greater than',
    (x, y) => le(y, x)
  )
}

export function toBeAllLessNotClose(
  this: { isNot: boolean },
  result: TensorLike | Tensor,
  expect: TensorLike | Tensor,
  params: { rtol?: number; atol?: number; broadcast?: boolean } = {}
) {
  return toBeAll.call(
    this,
    result,
    expect,
    params,
    'less than not close to',
    isLessNotClose(params)
  )
}

export function toBeAllGreaterNotClose(
  this: { isNot: boolean },
  result: TensorLike | Tensor,
  expect: TensorLike | Tensor,
  params: { rtol?: number; atol?: number; broadcast?: boolean } = {}
) {
  const le = isLessNotClose(params)
  return toBeAll.call(
    this,
    result,
    expect,
    params,
    'greater than not close to',
    (x, y) => le(y, x)
  )
}

expect.extend({
  toBeAllCloseTo,
  toBeAllLessOrClose,
  toBeAllLessNotClose,
  toBeAllGreaterOrClose,
  toBeAllGreaterNotClose
})
