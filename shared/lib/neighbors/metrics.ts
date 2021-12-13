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

export interface Metric<T> {
  (x: T, y: T): number
}

/**
 * Returns the Minkowski distance with the given power `p`.
 * It is equivalent to the p-norm of the absolute difference
 * between two vector.
 *
 * @param p The power/exponent of the Minkowski distance.
 * @returns `(u,v) => sum[i]( |u[i]-v[i]|**p ) ** (1/p)`
 */
export const minkowskiDistance = (p: number) => {
  if (p === Infinity) return chebyshevDistance

  if (!(p >= 1))
    // <- handles NaN
    throw new Error('minkowskiDistance(p): p must be >= 1.')

  const metric = (u: ArrayLike<number>, v: ArrayLike<number>) => {
    if (u.length != v.length)
      throw new Error(
        `minkowskiDistance(${p})(u,v): u and v must have same length.`
      )

    // Implementation
    // --------------
    // The implementation is based on Mozilla's polyfill for `Math.hypot`.
    // The idea is to scale the vector difference during summation, such that
    // the largest summand is 1. This avoids under- and overflow of exponentiation.
    // The final distance result is then scaled back up.
    //
    // References
    // ----------
    //   ..[1] https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/hypot
    let sum = 0,
      max = 0

    for (let i = u.length; i-- > 0; ) {
      let d = Math.abs(u[i] - v[i])
      if (isNaN(d)) return NaN
      if (d !== 0) {
        if (d > max) sum *= (max / (max = d)) ** p
        sum += (d / max) ** p
      }
    }

    return isFinite(max) ? sum ** (1 / p) * max : max
  }

  Object.defineProperty(metric, 'name', {
    value: `minkowskiDistance(${p})`,
    writable: false
  })
  return metric
}

/**
 * The Chebyshev distance metric, equivalent to `minkowskiDistance(Infinity)` or `(u,v) => sum(abs(u-v))`.
 */
export const chebyshevDistance = (
  u: ArrayLike<number>,
  v: ArrayLike<number>
) => {
  if (u.length != v.length)
    throw new Error('chebyshevDistance(x,y): x and y must have same length.')

  // underflow-safe p-norm, inspired by polyfill described in:
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/hypot
  let max = 0

  for (let i = u.length; i-- > 0; ) max = Math.max(max, Math.abs(u[i] - v[i]))

  return max
}
