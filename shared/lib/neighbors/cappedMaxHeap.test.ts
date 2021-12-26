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

import * as fc from 'fast-check'

import { CappedMaxHeap } from './cappedMaxHeap'
import { describe, it } from 'mocha'
import { assert } from 'chai'

describe('CappedMaxHeap', () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any

  const anyFloat = () => fc
    .double(-(2 ** 16), +(2 ** 16))
    .map((x) => Math.fround(x))

  const anyFloatArray = () =>
    fc.array<number>(anyFloat(), {
      minLength: 1,
      maxLength: 256
    })

  const anyInput = () =>
    anyFloatArray().chain<[number[], number]>( (arr) =>
      fc.nat(arr.length - 1).map( (k) => [arr, k + 1] )
    )

  it(`CappedMaxHeap() retains and sorts smallest k values`, () => {
    fc.assert(
      fc.property(anyInput(), ([values, k]) => {
        Object.freeze(values)
        const keys = new Float32Array(k)
        const vals = new Int32Array(k)

        let reference = Float32Array.from(values)
        reference.sort((x, y) => x - y)
        reference = reference.slice(0, k)

        const heap = new CappedMaxHeap(keys, vals)
        values.forEach((x, i) => heap.add(x, i))
        heap.sort()

        assert.deepEqual( keys, reference )
        for (let i = 0; i < k; i++) {
          assert.equal(keys[i], values[vals[i]])
        }
        assert.equal(new Set(vals).size, k)
      }),
      { numRuns: 256 }
    )
  })
})
