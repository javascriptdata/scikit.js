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

const ARITY = 8
const child = (parent: number) => parent * ARITY + 1
const parent = (child: number) => ((child - 1) / ARITY) | 0

/**
 * A priority queue using a derivation of a traditional binary heap
 * as data structure. In a binary heap a branch can have up to
 * two children. In an n-ary heap a branch can have n children.
 * Other than that, the same heap properties apply and heap
 * building and minimum-extraction is almost identical.
 *
 * @see {@link https://en.wikipedia.org/wiki/Priority_queue}
 */
export class NAryHeap<T> {
  private _heap: T[] = []
  private _compareFn: (x: T, y: T) => -1 | 0 | 1

  /**
   * Creates a new empty n-ary heap priority queue using the
   * given comparator function.
   *
   * @param compareFn The comparator function. `compareFn(x,y) == -1`
   */
  constructor(compareFn: (x: T, y: T) => -1 | 0 | 1) {
    this._compareFn = compareFn
  }

  get size() {
    return this._heap.length
  }

  /**
   * Enqueue a new element in this n-ary heap priority queue.
   *
   * @param item The item to be enqueued.
   */
  add(item: T) {
    const { _heap, _compareFn } = this
    _heap.push(item)

    // SIFT UP
    let i = _heap.length - 1
    for (;;) {
      const p = parent(i)
      if (i <= 0 || _compareFn(item, _heap[p]) >= 0) break
      _heap[i] = _heap[(i = p)]
    }
    _heap[i] = item
  }

  /**
   * Peeks at minimum element without removing it.
   */
  get min() {
    return this._heap[0]
  }

  /**
   * Removes the minimum element and returns it.
   *
   * @returns The minimum element which has been removed or undefined if empty.
   */
  popMin() {
    const { _heap, _compareFn } = this,
      result = _heap[0],
      filler = _heap.pop() as T, // <- use rightmost entry to fill gap
      len = _heap.length
    if (len > 0) {
      // sift-down the root
      for (let i = 0; ; ) {
        const p = i
        // fill parental gap with filler
        _heap[i] = filler
        let c = child(i)

        // find largest child that is larger than the filler
        let lastChild = c + ARITY
        if (lastChild > len) lastChild = len

        for (; c < lastChild; c++)
          if (_compareFn(_heap[c], _heap[i]) < 0) i = c

        // if parent is already the smallest value, stop
        if (i === p) break

        // move smallest value to root/parent
        _heap[p] = _heap[i]
      }
    }

    return result
  }
}
