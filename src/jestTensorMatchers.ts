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

import { getBackend } from './tf-singleton'
import { Tensor, TensorLike } from './types'
import { isTensor } from './typesUtils'

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace jest {
    interface Matchers<R, T> {
      /**
       * Tests whether or not every entry of a Tensor(Like) is close
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
            params?: {
              rtol?: number
              atol?: number
              broadcast?: boolean
              allowEmpty?: boolean
            }
          ) => R
        : undefined
      /**
       * Tests whether or not every entry of a Tensor(Like) is less than or close
       * to the corresponding entry of an expected tensor.
       *
       * @param expected The expected result tensor.
       * @param params Tolerance and broadcast settings. `{broadcast: false}` disallows
       *               broadcasting, i.e. the result tensor must have the same shape as
       *               the expected tensor. `{rtol, atol}` are the relative and absolute
       *               tolerance parameter. Two entries `x` and `y` are considered equal
       *               iff `x-y <= max(|x|, |y|)*rtol + atol`. Set `{rtol: 0, atol: 0}`
       *               for exact equality.
       */
      toBeAllLessOrClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: {
              rtol?: number
              atol?: number
              broadcast?: boolean
              allowEmpty?: boolean
            }
          ) => R
        : undefined
      /**
       * Tests whether or not every entry of a Tensor(Like) is greater than or close
       * to the corresponding entry of an expected tensor.
       *
       * @param expected The expected result tensor.
       * @param params Tolerance and broadcast settings. `{broadcast: false}` disallows
       *               broadcasting, i.e. the result tensor must have the same shape as
       *               the expected tensor. `{rtol, atol}` are the relative and absolute
       *               tolerance parameter. Two entries `x` and `y` are considered equal
       *               iff `x-y >= -max(|x|, |y|)*rtol - atol`. Set `{rtol: 0, atol: 0}`
       *               for exact equality.
       */
      toBeAllGreaterOrClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: {
              rtol?: number
              atol?: number
              broadcast?: boolean
              allowEmpty?: boolean
            }
          ) => R
        : undefined
      /**
       * Tests whether or not every entry of a Tensor(Like) is sufficiently less than
       * the corresponding entry of an expected tensor.
       *
       * @param expected The expected result tensor.
       * @param params Tolerance and broadcast settings. `{broadcast: false}` disallows
       *               broadcasting, i.e. the result tensor must have the same shape as
       *               the expected tensor. `{rtol, atol}` are the relative and absolute
       *               tolerance parameter. Two entries `x` and `y` are considered equal
       *               iff `x-y < -max(|x|, |y|)*rtol - atol`. Set `{rtol: 0, atol: 0}`
       *               for exact equality.
       */
      toBeAllLessNotClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: {
              rtol?: number
              atol?: number
              broadcast?: boolean
              allowEmpty?: boolean
            }
          ) => R
        : undefined
      /**
       * Tests whether or not every entry of a Tensor(Like) is sufficiently greater than
       * the corresponding entry of an expected tensor.
       *
       * @param expected The expected result tensor.
       * @param params Tolerance and broadcast settings. `{broadcast: false}` disallows
       *               broadcasting, i.e. the result tensor must have the same shape as
       *               the expected tensor. `{rtol, atol}` are the relative and absolute
       *               tolerance parameter. Two entries `x` and `y` are considered equal
       *               iff `x-y > max(|x|, |y|)*rtol + atol`. Set `{rtol: 0, atol: 0}`
       *               for exact equality.
       */
      toBeAllGreaterNotClose: T extends Tensor | TensorLike
        ? (
            expected: Tensor | TensorLike,
            params?: {
              rtol?: number
              atol?: number
              broadcast?: boolean
              allowEmpty?: boolean
            }
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
  { broadcast = true, allowEmpty = false },
  description: string,
  match: (x: number, y: number) => boolean
) {
  let tf = getBackend()
  const { isNot } = this
  const a = isTensor(result) ? result : tf.tensor(result)
  const b = isTensor(expect) ? expect : tf.tensor(expect)

  const msg = (msg: string) => () =>
    `\nA: ${a.toString(true)}` +
    `\nB: ${b.toString(true)}` +
    `\nExpected A ${
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
  // flattened data
  const aFlat = a.dataSync()
  const bFlat = b.dataSync()

  if (aFlat.length === 0 || bFlat.length === 0) {
    return {
      pass: allowEmpty !== isNot,
      message: msg('Empty shape(s) encountered.')
    }
  }

  // indices into flattened data
  let ia = 0
  let ib = 0

  // inside of `visit(axis)`, stride counts amount of
  // element that had been visited by a call to
  // `visit(axis+1)`. Used to repeat elements along
  // axis in case of broadcasting
  let strideA: number
  let strideB: number

  /* Visits broadcasted pairs of entries. Needs
   * to be recursive to allow for arbitrary ranks.
   */
  function visit(axis: number) {
    if (axis === rank) {
      if (!match(aFlat[ia], bFlat[ib])) {
        throw msg(
          `A[${unravelIndex(ia, shapeA)}] = ${aFlat[ia]}\n` +
            `B[${unravelIndex(ib, shapeB)}] = ${bFlat[ib]}`
        )
      }
      strideA = 1
      strideB = 1
      ia++
      ib++
    } else {
      for (let i = 0; ; ) {
        visit(axis + 1)
        if (++i >= shape[axis]) {
          break
        }
        // Broadcasting cases, repeat entries alond axis.
        // Utilizes fact that `shape[i < 0] === undefined`.
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
