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

import { Tensor, Tensor2D } from '@tensorflow/tfjs'

import { tf } from '../../globals'
import { assert } from '../typesUtils'

/**
 * Abstract type of neighbohood distance metrics.
 */
export interface Metric {
  /**
   * Returns the broadcasted distances between a batch of points `X` and another
   * batch points `Y`.
   *
   * @param X One batch of points, where the last axis represents the point
   *          coordinates. The leading axes represent the batch dimensions.
   * @param Y Other batch of points, where the last axis represents the point
   *          coordinates. The leading axes represent the batch dimensions.
   *
   * @returns A broadcasted distance tensor `D`, where `D[..., i, j]` represents
   *          the distance between point `X[..., i, j, k]` and point `Y[..., i, j, k]`.
   */
  tensorDistance(u: Tensor, v: Tensor): Tensor

  /**
   * Returns the distance between two points `u` and `v`.
   *
   * @param u The 1st point coordinates. Must have same length as `v`.
   * @param v The 2nd point coordinates. Must have same length as `u`.
   *
   * @return The distance between `u` and `v` according to this
   *         metric.
   */
  distance(u: ArrayLike<number>, v: ArrayLike<number>): number

  /**
   * Returns minimum distance of a point to a bounding box.
   *
   * @param pt The point coordinates. Must be half as long as `bBox`.
   * @param bBox The bounding box bounds, where `bBox[2*i]` is the lower
   *             bound of coordinate `i` and `bBox[2*i+1]` is the upper
   *             bound of coordinate `i`.
   */
  minDistToBBox?(pt: ArrayLike<number>, bBox: ArrayLike<number>): number

  /**
   * Name of the metric.
   */
  name: string

  /**
   * Returns the name of the metric.
   */
  toString(): string
}

const minkowskiTensorDistance = (p: number) => (u: Tensor, v: Tensor) => {
  // FIXME: tf.norm still underflows and overflows,
  // see: https://github.com/tensorflow/tfjs/issues/895
  const m = u.shape[u.rank - 1] ?? NaN
  const n = v.shape[v.rank - 1] ?? NaN
  assert(
    m === n,
    `minkowskiDistance(${p}).tensorDistance(u,v): u.shape[-1] must equal v.shape[-1].`
  )
  return tf.tidy(() => {
    return tf.norm(tf.sub(u, v), p, -1)
  }) as Tensor2D
}

/**
 * Returns the Minkowski distance metric with the given power `p`.
 * It is equivalent to the p-norm of the absolute difference
 * between two vectors.
 *
 * @param p The power/exponent of the Minkowski distance.
 * @returns `(X,y) => sum[i]( |X[:,i]-y[i]|**p ) ** (1/p)`
 */
export const minkowskiMetric = (p: number) => {
  switch (p) {
    case 1:
      return manhattanMetric
    case 2:
      return euclideanMetric
    case Infinity:
      return chebyshevMetric
  }
  assert(1 <= p, 'minkowskiMetric(p): Invalid p.')

  const metric = {
    tensorDistance: minkowskiTensorDistance(p),
    distance(u: ArrayLike<number>, v: ArrayLike<number>) {
      const len = u.length
      if (len !== v.length) {
        throw new Error(
          `minkowskiMetric(${p}).treeMetric(u,v): u and v must have same length.`
        )
      }
      // since we are aming at float32 precision, this
      // implementation is not underflow-/ overflow-safe
      // TODO: if tfjs ever adds float64, make this underflow-safe
      let norm = 0
      for (let i = 0; i < len; i++) {
        norm += Math.abs(u[i] - v[i]) ** p
      }
      return norm ** (1 / p)
    },
    distToBBox(pt: ArrayLike<number>, bBox: ArrayLike<number>) {
      if (pt.length * 2 != bBox.length) {
        throw new Error(
          `minkowskiMetric(${p}).treeMetric.minDistToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
        )
      }
      let norm = 0
      for (let j = 0, i = 0; i < pt.length; i++) {
        let x = Math.max(0, bBox[j++] - pt[i], pt[i] - bBox[j++])
        norm += x ** p
      }
      return norm ** (1 / p)
    },
    name: `minkowskiMetric(${p})`,
    toString() {
      return this.name
    }
  }

  return Object.freeze(metric) as Metric
}

const manhattanMetric: Metric = Object.freeze({
  tensorDistance: minkowskiTensorDistance(1),
  distance(u: ArrayLike<number>, v: ArrayLike<number>) {
    const len = u.length
    if (len !== v.length) {
      throw new Error(
        `minkowskiMetric(1).distance(u,v): u and v must have same length.`
      )
    }
    let norm = 0
    for (let i = 0; i < len; i++) {
      norm += Math.abs(u[i] - v[i])
    }
    return norm
  },
  minDistToBBox(pt: ArrayLike<number>, bBox: ArrayLike<number>) {
    const len = bBox.length
    if (len !== pt.length << 1) {
      throw new Error(
        `minkowskiMetric(1).minDistToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
      )
    }
    let norm = 0
    for (let i = 0; i < len; ) {
      const pi = pt[i >>> 1]
      // const x = Math.max(0, bBox[i++] - pi, pi - bBox[i++])
      const u = bBox[i++] - pi
      const v = pi - bBox[i++]
      const x = 0.5 * (Math.abs(u) + u + (Math.abs(v) + v))
      norm += x
    }
    return norm
  },
  name: 'manhattanMetric',
  toString() {
    return this.name
  }
})

const euclideanMetric: Metric = Object.freeze({
  tensorDistance: minkowskiTensorDistance(2),
  distance(u: ArrayLike<number>, v: ArrayLike<number>) {
    const len = u.length
    if (len !== v.length) {
      throw new Error(
        `minkowskiMetric(2).distance(u,v): u and v must have same length.`
      )
    }
    let norm = 0
    for (let i = 0; i < len; i++) {
      const x = u[i] - v[i]
      norm += x * x
    }
    return Math.sqrt(norm)
  },
  minDistToBBox(pt: ArrayLike<number>, bBox: ArrayLike<number>) {
    const len = bBox.length
    if (len !== pt.length * 2) {
      throw new Error(
        `minkowskiMetric(2).minDistToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
      )
    }
    let norm = 0
    for (let i = 0; i < len; ) {
      const pi = pt[i >>> 1]
      // const x = Math.max(0, bBox[i++] - pi, pi - bBox[i++])
      const u = bBox[i++] - pi
      const v = pi - bBox[i++]
      const x = 0.5 * (Math.abs(u) + u + (Math.abs(v) + v))
      norm += x * x
    }
    return Math.sqrt(norm)
  },
  name: 'euclideanMetric',
  toString() {
    return this.name
  }
})

const chebyshevMetric: Metric = Object.freeze({
  tensorDistance: minkowskiTensorDistance(Infinity),
  distance(u: ArrayLike<number>, v: ArrayLike<number>) {
    const len = u.length
    if (len !== v.length) {
      throw new Error(
        `minkowskiMetric(Infinity).distance(u,v): u and v must have same length.`
      )
    }
    let norm = 0
    for (let i = 0; i < len; i++) {
      const x = Math.abs(u[i] - v[i])
      norm = Math.max(norm, x)
    }
    return norm
  },
  minDistToBBox(pt: ArrayLike<number>, bBox: ArrayLike<number>) {
    const len = bBox.length
    if (len !== pt.length * 2) {
      throw new Error(
        `minkowskiMetric(Infinity).minDistToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
      )
    }
    let norm = -Infinity
    for (let i = 0; i < len; ) {
      const pi = pt[i >>> 1]
      // const x = Math.max(0, bBox[i++] - pi, pi - bBox[i++])
      const u = bBox[i++] - pi
      const v = pi - bBox[i++]
      const x = 0.5 * (Math.abs(u) + u + (Math.abs(v) + v))
      norm = Math.max(norm, x)
    }
    return norm
  },
  name: 'chebyshevMetric',
  toString() {
    return this.name
  }
})
