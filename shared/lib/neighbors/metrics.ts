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

import { Tensor2D } from '@tensorflow/tfjs'

import { tf } from '../../globals'
import { assert } from '../typesUtils'

type Vec = ArrayLike<number>

export interface TreeMetric {
  (u: Vec, v: Vec): number
  distToBBox( point: Vec, bBox: Vec ): number
}

/**
 * Abstract type of distance metrics, which compute the
 * distances between a stack of points `X` and a single point `y`.
 *
 * @param X A 2d tensor where each row represents a point.
 * @param y A 2d tensor where each row represents a point.
 *
 * @returns A 2d tensor `dist` where `dist[i,j]` is the
 *          metric distance between `u[i,:]` and `v[j,:]`.
 */
export interface Metric {
  (u: Tensor2D, v: Tensor2D): Tensor2D
  treeMetric: TreeMetric
}

/**
 * Returns the Minkowski distance metric with the given power `p`.
 * It is equivalent to the p-norm of the absolute difference
 * between two vectors.
 *
 * @param p The power/exponent of the Minkowski distance.
 * @returns `(X,y) => sum[i]( |X[:,i]-y[i]|**p ) ** (1/p)`
 */
export const minkowskiDistance = (p: number) => {
  const metric = (u: Tensor2D, v: Tensor2D) => {
    // FIXME: tf.norm still underflows and overflows,
    // see: https://github.com/tensorflow/tfjs/issues/895
    const [m, s] = u.shape
    const [n, t] = v.shape
    assert(
      s == t,
      `minkowskiDistance(${p})(u,v): u.shape[1] must equal v.shape[1].`
    )
    return tf.tidy(() => {
      const x = u.reshape([m, 1, s])
      const y = v.reshape([1, n, t])
      return tf.norm(tf.sub(x, y), p, -1)
    }) as Tensor2D
  }

  metric.treeMetric = minkowskiTreeMetric(p)
  Object.defineProperty(metric, 'name', { value: `minkowskiDistance(${p})`, writable: true })
  return metric as Metric
}

const minkowskiTreeMetric = (p: number) => {
  switch (p) {
    case 1: return manhattanTreeMetric
    case 2: return euclideanTreeMetric
    case Infinity: return chebyshevTreeMetric
  }

  assert(1 <= p, 'minkowskiDistance(p): Invalid p.')

  const treeMetric = (u: Vec, v: Vec) => {
    const len = u.length
    if (len !== v.length) {
      throw new Error(
        `minkowskiDistance(${p}).treeMetric(u,v): u and v must have same length.`
      )
    }

    let norm = 0

    for (let i = 0; i < len; i++ ) {
      norm += Math.abs(u[i] - v[i]) ** p
    }

    return norm ** (1 / p)
  }

  treeMetric.distToBBox = (pt: Vec, bBox: Vec) => {
    if (pt.length * 2 != bBox.length) {
      throw new Error(
        `minkowskiDistance(${p}).treeMetric.distToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
      )
    }

    let norm = 0

    for (let j = 0, i = 0; i < pt.length; i++ ) {
      let x = Math.max(0, bBox[j++] - pt[i], pt[i] - bBox[j++])
      norm += x ** p
    }

    return norm ** (1 / p)
  }

  Object.defineProperty(treeMetric, 'name', { value: `minkowskiDistance(${p}).treeMetric`, writable: true })
  return treeMetric as TreeMetric
}

const manhattanTreeMetric = (u: Vec, v: Vec) => {
  const len = u.length
  if (len !== v.length) {
    throw new Error(
      `minkowskiDistance(1).treeMetric(u,v): u and v must have same length.`
    )
  }

  let norm = 0

  for (let i = 0; i < len; i++ ) {
    norm += Math.abs(u[i] - v[i])
  }

  return norm
}
manhattanTreeMetric.distToBBox = (pt: Vec, bBox: Vec) => {
  const len = bBox.length
  if (len !== (pt.length << 1)) {
    throw new Error(
      `minkowskiDistance(1).treeMetric.distToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
    )
  }

  let norm = 0

  for (let i = 0; i < len; ) {
    const pi = pt[i >>> 1]

//    const x = Math.max(0, bBox[i++] - pi, pi - bBox[i++])

    const u = bBox[i++] - pi
    const v = pi - bBox[i++]
    const x = 0.5 * ((Math.abs(u) + u) + (Math.abs(v) + v))

    norm += x
  }

  return norm
}
Object.defineProperty(manhattanTreeMetric, 'name', { value: `minkowskiDistance(1).treeMetric`, writable: true })

const euclideanTreeMetric = (u: Vec, v: Vec) => {
  const len = u.length
  if (len !== v.length) {
    throw new Error(
      `minkowskiDistance(2).treeMetric(u,v): u and v must have same length.`
    )
  }

  let norm = 0

  for (let i = 0; i < len; i++ ) {
    const x = u[i] - v[i]
    norm += x * x
  }

  return Math.sqrt(norm)
}
euclideanTreeMetric.distToBBox = (pt: Vec, bBox: Vec) => {
  const len = bBox.length
  if (pt.length * 2 !== bBox.length) {
    throw new Error(
      `minkowskiDistance(2).treeMetric.distToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
    )
  }

  let norm = 0

  for (let i = 0; i < len; ) {
    const pi = pt[i >>> 1]

//    const x = Math.max(0, bBox[i++] - pi, pi - bBox[i++])

    const u = bBox[i++] - pi
    const v = pi - bBox[i++]
    const x = 0.5 * ((Math.abs(u) + u) + (Math.abs(v) + v))
    norm += x * x
  }

  return Math.sqrt(norm)
}
Object.defineProperty(euclideanTreeMetric, 'name', { value: `minkowskiDistance(2).treeMetric`, writable: true })

const chebyshevTreeMetric = (u: Vec, v: Vec) => {
  const len = u.length
  if (len !== v.length) {
    throw new Error(
      `minkowskiDistance(Infinity).treeMetric(u,v): u and v must have same length.`
    )
  }

  let norm = -Infinity

  for (let i = 0; i < len; i++ ) {
    const x = Math.abs(u[i] - v[i])
    norm = Math.max(norm, x)
  }

  return norm
}
chebyshevTreeMetric.distToBBox = (pt: Vec, bBox: Vec) => {
  const len = bBox.length
  if (pt.length * 2 != bBox.length) {
    throw new Error(
      `minkowskiDistance(Infinity).treeMetric.distToBBox(pt,bBox): pt.length*2 must equal bBox.length.`
    )
  }

  let norm = -Infinity

  for (let i = 0; i < len;) {
    const pi = pt[i >>> 1]

//    const x = Math.max(0, bBox[i++] - pi, pi - bBox[i++])

    const u = bBox[i++] - pi
    const v = pi - bBox[i++]
    const x = 0.5 * ((Math.abs(u) + u) + (Math.abs(v) + v))
    norm = Math.max(norm, x)
  }

  return norm
}
Object.defineProperty(chebyshevTreeMetric, 'name', { value: `minkowskiDistance(Infinity).treeMetric`, writable: true })
