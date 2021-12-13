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
import { alea } from 'seedrandom'

import { lhs } from '../randUtils'

import { Metric, minkowskiDistance } from './metrics'
import { Neighborhood, NeighborhoodParams } from './neighborhood'

export const run_generic_neighborhood_tests = (
  NeighborhoodImpl: new <A, V>(
    params: NeighborhoodParams<A, V>
  ) => Neighborhood<A, V>
) => {
  describe(`${NeighborhoodImpl.name} [generic tests]`, () => {
    const metrics = {
      'p-norm(p=1)': minkowskiDistance(1),
      'p-norm(p=2)': minkowskiDistance(2),
      'p-norm(p=3)': minkowskiDistance(3),
      'p-norm(p=Inf)': minkowskiDistance(Infinity)
    }

    const methods: ('nearest' | 'farthest')[] = ['nearest', 'farthest']

    const anyAddress = (ndim: number) => {
      // small value to provoke underflow (nextDown(x)**2 === 0)
      const small = 1.5717277847026288e-162

      // large value to provoke overflow (x*x === Infinity)
      const large = 1.3407807929942597e154

      const scale = 2 ** 16

      const anyDouble = fc.oneof(
        fc.double(-scale, +scale),
        fc.oneof(
          fc.double(0, +small * scale), // <- underflow
          fc.double(-small * scale, 0), // <- underflow
          fc.double(large / scale, +Number.MAX_VALUE), // <- overflow
          fc.double(-Number.MAX_VALUE, large / scale) // <- overflow
        )
      )

      return fc.array(
        fc.array(anyDouble, { minLength: ndim, maxLength: ndim })
      )
    }

    const testNeighbohood = (
      method: 'nearest' | 'farthest',
      metric: Metric<ArrayLike<number>>,
      entryAddr: number[][],
      searchAddr: number[][]
    ) => {
      const entries = Object.freeze(
        entryAddr.map((address, value) => ({ address, value }))
      )

      const hood = new NeighborhoodImpl({ metric, entries }),
        visited = entries.map(() => false)

      for (const searchAddress of searchAddr) {
        visited.fill(false)

        let previousDistance = 'nearest' === method ? 0 : Infinity
        for (const { address, distance, value } of hood[method](
          searchAddress
        )) {
          // make sure distance is computed correctly
          assert.equal(distance, metric(searchAddress, address))

          // make sure distances are ordered
          if ('nearest' === method)
            assert.isAtLeast(distance, previousDistance)
          else assert.isAtMost(distance, previousDistance)
          previousDistance = distance

          // make sure each entry is visited only once
          assert.isFalse(visited[value])
          visited[value] = true

          // make sure the entry has right address
          assert.deepEqual(address, entries[value].address)
        }

        // make sure all entries have been yielded
        assert.isTrue(visited.every((x) => x))
      }
    }

    for (const method of methods)
      for (const metric of Object.values(metrics))
        it(`*${method}(x) yields correct entries searching for entry addresses [${metric.name}]`, () => {
          fc.assert(
            fc.property(
              fc.nat(8).chain((ndim) => anyAddress(++ndim)),
              (entryAddr) =>
                testNeighbohood(method, metric, entryAddr, entryAddr)
            )
          )
        })

    for (const method of methods)
      for (const metric of Object.values(metrics))
        it(`*${method}(x) yields correct entries searching for other addresses [${metric.name}]`, () => {
          fc.assert(
            fc.property(
              fc
                .nat(8)
                .map((ndim) => ++ndim)
                .chain((ndim) => fc.tuple(anyAddress(ndim), anyAddress(ndim))),
              ([entryAddr, searchAddr]) =>
                testNeighbohood(method, metric, entryAddr, searchAddr)
            )
          )
        })

    for (const metric of Object.values(metrics))
      it(`*nearest(x) starts with distinct entries x [${metric.name}]`, () => {
        const anyEntries = fc
          .tuple(fc.string(), fc.nat(128), fc.nat(4))
          .map(([seed, nSamples, nFeatures]) => {
            return lhs(alea(seed))(nSamples + 1, nFeatures + 1).map(
              (address, value) => ({ address, value })
            )
          })

        fc.assert(
          fc.property(anyEntries, (entries) => {
            Object.freeze(entries)
            const hood = new NeighborhoodImpl({ metric, entries })

            for (const {
              address: searchAddress,
              value: searchValue
            } of entries) {
              const { address, distance, value } = hood
                .nearest(searchAddress)
                .next().value
              assert.equal(distance, 0)
              assert.equal(value, searchValue)
              assert.deepEqual(address, searchAddress)
            }
          })
        )
      })
  })
}
