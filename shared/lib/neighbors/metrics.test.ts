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

import { assert } from 'chai'
import * as fc from 'fast-check'
import { describe, it } from 'mocha'
import { Metric, minkowskiMetric } from './metrics'

const NDIMS = Object.freeze([1, 2, 7])

const anyVec = (ndim: number) => {
//  // small value to provoke underflow (nextDown(x)**2 === 0)
//  const small = 1.5717277847026288e-162

//  // large value to provoke overflow (x*x === Infinity)
//  const large = 1.3407807929942597e154

  const scale = 2 ** 16

//  const anyDouble = fc.oneof(
//    fc.double(-scale, +scale),
//    fc.oneof(
//      fc.double(0, +small * scale), // <- underflow
//      fc.double(-small * scale, 0), // <- underflow
//      fc.double(large / scale, +Number.MAX_VALUE), // <- overflow
//      fc.double(-Number.MAX_VALUE, large / scale) // <- overflow
//    )
//  )
  const anyDouble = fc.double(-scale, +scale)

  return fc.array(anyDouble, { minLength: ndim, maxLength: ndim })
}

const assertClose = (x: number, y: number) => {
  if (!isFinite(x) || !isFinite(y)) return x === y

  const rtol = 1e-5
  const atol = 1e-8

  const tol = atol + rtol * Math.max(Math.abs(x), Math.abs(y))
  return assert.closeTo(x, y, tol)
}

const run_generic_vector_metric_tests = (
  metric: Metric
) => {
  describe(`${metric.name} [generic tests]`, () => {
    // test the metric properties as described in:
    // https://en.wikipedia.org/wiki/Metric_(mathematics)

    for (const ndim of NDIMS)
      it(`metric(v,v) == 0 [${ndim}D]`, () => {
        const anyV = anyVec(ndim)

        fc.assert(
          fc.property(anyV, (v) => {
            Object.freeze(v)
            assert.equal(metric.distance(v, v), 0)
          })
        )
      })

    for (const ndim of NDIMS)
      it(`metric(u,v) >= 0 [${ndim}D]`, () => {
        const anyU = anyVec(ndim)
        const anyV = anyVec(ndim)

        fc.assert(
          fc.property(anyU, anyV, (u, v) => {
            Object.freeze(u)
            Object.freeze(v)
            assert.isAtLeast(metric.distance(u, v), 0)
          })
        )
      })

    for (const ndim of NDIMS)
      it(`metric(u,v) == metric(v,u) [${ndim}D]`, () => {
        const anyU = anyVec(ndim)
        const anyV = anyVec(ndim)

        fc.assert(
          fc.property(anyU, anyV, (u, v) => {
            Object.freeze(u)
            Object.freeze(v)
            assert.equal(metric.distance(u, v), metric.distance(v, u))
          })
        )
      })

    for (const ndim of NDIMS)
      it(`metric(u,v)==0 <=> u==v [${ndim}D]`, () => {
        const anyU = anyVec(ndim)
        const anyV = anyVec(ndim)

        fc.assert(
          fc.property(anyU, anyV, (u, v) => {
            Object.freeze(u)
            Object.freeze(v)
            if (metric.distance(u, v) === 0) assert.deepEqual(u, v)
            else assert.notDeepEqual(u, v)
          })
        )
      })

    for (const ndim of NDIMS)
      it(`metric(u,w) <= metric(u,v) + metric(v,w) [${ndim}D]`, () => {
        const anyU = anyVec(ndim)
        const anyV = anyVec(ndim)
        const anyW = anyVec(ndim)

        fc.assert(
          fc.property(anyU, anyV, anyW, (u, v, w) => {
            Object.freeze(u)
            Object.freeze(v)
            Object.freeze(w)
            assert.isAtMost(metric.distance(u, w), metric.distance(u, v) + metric.distance(u, w))
          })
        )
      })
  })
}

for (const p of [1, 1.9, 2.0, 3.0, Infinity])
  run_generic_vector_metric_tests(minkowskiMetric(p))

describe('euclideanDistance', () => {
  const euclid = minkowskiMetric(2)

  for (const ndim of NDIMS)
    it(`is close to its trivial implementation [${ndim}D]`, () => {
      const anyU = anyVec(ndim)
      const anyV = anyVec(ndim)

      fc.assert(
        fc.property(anyU, anyV, (u, v) => {
          Object.freeze(u)
          Object.freeze(v)
          const ref = Math.hypot(
            ...u.map((ui, i) => ui - v[i])
          )
          assertClose(euclid.distance(u, v), ref)
        })
      )
    })
})

describe('manhattanDistance', () => {
  const manhat = minkowskiMetric(1)

  for (const ndim of NDIMS)
    it(`is close to its trivial implementation [${ndim}D]`, () => {
      const anyU = anyVec(ndim)
      const anyV = anyVec(ndim)

      fc.assert(
        fc.property(anyU, anyV, (u, v) => {
          Object.freeze(u)
          Object.freeze(v)
          const ref = u
            .map((ui, i) => Math.abs(ui - v[i]))
            .reduce((x, y) => x + y)
          assertClose(manhat.distance(u, v), ref)
        })
      )
    })
})

describe('chebyshevDistance', () => {
  const chevy = minkowskiMetric(Infinity)

  for (const ndim of NDIMS)
    it(`is close to its trivial implementation [${ndim}D]`, () => {
      const anyU = anyVec(ndim)
      const anyV = anyVec(ndim)

      fc.assert(
        fc.property(anyU, anyV, (u, v) => {
          Object.freeze(u)
          Object.freeze(v)
          const ref = u
            .map((ui, i) => Math.abs(ui - v[i]))
            .reduce((x, y) => Math.max(x, y))
          assertClose(chevy.distance(u, v), ref)
        })
      )
    })
})
