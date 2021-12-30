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

import { NAryHeap } from './nAryHeap'
import { describe, it } from 'mocha'
import { assert } from 'chai'

type int = number

describe('NAryHeap', () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any

  const anyFloat = () =>
    fc.double(-(2 ** 16), +(2 ** 16)).map((x) => Math.fround(x))

  const anyFloatArray = () =>
    fc.array<number>(anyFloat(), {
      maxLength: 512
    })

  it(`NAryHeap() returns items in order`, () => {
    fc.assert(
      fc.property(anyFloatArray(), (entries) => {
        const heap = new NAryHeap<int>()

        for (let i = 0; i < entries.length; i++) heap.add(entries[i], i)

        const unvisited = entries.map(() => true)

        let previous = -Infinity
        // make sure elements are returned in order
        for (let i = entries.length; i > 0; ) {
          assert.equal(heap.size, i)
          const key = heap.minKey
          const val = heap.minVal
          assert.isTrue(unvisited[val])
          unvisited[val] = false
          assert.equal(entries[val], key)
          assert.equal(heap.size, i)
          assert.equal(val, heap.popMin())
          assert.equal(heap.size, --i)
          assert.isAtMost(previous, (previous = key))
        }

        assert.deepEqual(
          unvisited,
          entries.map(() => false)
        )
      }),
      { numRuns: 512 }
    )
  })
})
