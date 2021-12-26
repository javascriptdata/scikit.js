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
 * @see {@link https://en.wikipedia.org/wiki/Binary_heap}
 */
export class NAryHeap<T> {
  private _keys = new Float32Array(8)
  private _vals: T[] = []

  constructor() {
    this._keys[0] = Infinity
  }

  get size() {
    return this._vals.length
  }

  /**
   * Enqueues a new entry in this n-ary heap priority queue.
   *
   * @param key Key to be enqueueud.
   * @param val Value to be enqueued.
   */
  add( key: number, val: T ) {
    let { _keys, _vals } = this
    let i = _vals.length
    if (i >= _keys.length) {
      _keys = new Float32Array(_keys.length << 1)
      _keys.set(this._keys)
      this._keys = _keys
    }
    _vals.push(val)

    // SIFT UP
    for (;;) {
      const p = parent(i)
      if (i <= 0 || !(key < _keys[p])) break
      _keys[i] = _keys[p]
      _vals[i] = _vals[p]
      i = p
    }
    _keys[i] = key
    _vals[i] = val
  }

  /**
   * Returns the minimum key.
   */
  get minKey() {
    return this._keys[0]
  }

  /**
   * Returns the minimum value.
   */
  get minVal() {
    return this._vals[0]
  }

  /**
   * Removes the minimum element without returning it.
   */
  popMin(): T {
    const { _keys, _vals } = this,
       ret = _vals[0],
       val = _vals.pop() as T,
      size = _vals.length,
       key = _keys[size]

    if (size > 0) {
      // sift-down the root
      for (let i = 0; ; ) {
        const p = i
        // fill parental gap with filler
        _keys[i] = key
        _vals[i] = val
        let c = child(i)

        // find largest child that is larger than the filler
        const lastChild = Math.min(size, c + ARITY)
        for (;lastChild > c; c++)
          if (_keys[c] < _keys[i]) i = c

        // if parent is already the smallest value, stop
        if (i === p) break

        // move smallest value to root/parent
        _keys[p] = _keys[i]
        _vals[p] = _vals[i]
      }
    }
    else {
      _keys[0] = Infinity
    }

    return ret
  }
}
