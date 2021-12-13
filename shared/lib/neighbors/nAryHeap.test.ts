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

describe('NAryHeap', () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function test_with(
    compareFn: (x: any, y: any) => -1 | 0 | 1,
    elemType: 'integer' | 'double' | 'string'
  ) {
    const test_property = (
      description: string,
      test_body: (arr: (string | number)[]) => void
    ) => {
      description = `${description} (${compareFn.name} order, ${elemType}[] inputs)`

      it(description, () => {
        const arrays = fc.array<string | number>(fc[elemType](), {
          maxLength: 100
        })
        fc.assert(fc.property(arrays, test_body), { numRuns: 100 })
      })
    }

    test_property('returns elements in sorted order', (arr) => {
      const heap = new NAryHeap(compareFn)

      for (const x of arr) heap.add(x)

      const ref = arr.slice()
      ref.sort((x, y) => -compareFn(x, y))

      // make sure elements are returned in order
      for (let i = ref.length; i-- > 0; ) {
        assert.equal(heap.size, i + 1)
        assert.equal(heap.min, ref[i])
        assert.equal(heap.popMin(), ref[i])
        assert.equal(heap.size, i)
      }
    })
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ascending = (x: any, y: any) => (x < y ? -1 : +1)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const descending = (x: any, y: any) => (x > y ? -1 : +1)

  for (const compareFn of [ascending, descending]) {
    test_with(compareFn, 'integer')
    test_with(compareFn, 'double')
    test_with(compareFn, 'string')
  }
})
