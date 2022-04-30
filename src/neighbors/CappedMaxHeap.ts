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

import { assert } from '../typesUtils'

/**
 * A variant of a binary max-heap with a size limit.
 * If more elements than the size limit are added,
 * only the smallest encountered elements are retained.
 * This data structure is used in k-nearest-neighbor
 * searches to retain the k nearest results during
 * tree traversal.
 */
export class CappedMaxHeap {
  _keys: Float32Array
  _vals: Int32Array
  /**
   * Index of the currently first entry.
   * The entries are added from right to
   * left until `_pos = 0`, at which point
   * adding further elements results in
   * replacement.
   */
  _pos: number

  /**
   * Creates a new heap using the given key
   * and value array as underlying data storage.
   * The heap will modify these arrays directly.
   *
   * @param keys Key array to be used by the heap.
   * @param vals Value array to be used by the heap.
   */
  constructor(keys: Float32Array, vals: Int32Array) {
    const len = keys.length
    assert(
      len > 0,
      'new CappedMaxHeap(keys,vals): keys.length must be positive.'
    )
    assert(
      len === vals.length,
      'new CappedMaxHeap(keys,vals): keys.length must equal vals.length.'
    )
    keys[0] = NaN
    this._keys = keys
    this._vals = vals
    this._pos = len
  }

  /**
   * Returns the currently larges key without removing it.
   */
  get maxKey() {
    return this._keys[0]
  }

  /**
   * Adds an entry to this heap. If, after adding,
   * the size limit is exceeded, the largest entry
   * is removed as well, even the entry that was
   * just added if it has the largest key.
   *
   * @param key Key of the added entry.
   * @param val Value of the added entry.
   */
  add(key: number, val: number): void {
    let { _keys, _vals, _pos: p } = this
    if (0 < p) {
      this._pos = --p
    } else if (_keys[0] <= key) {
      return
    }
    const end = _keys.length - 1

    // sift-down
    for (;;) {
      let c = (p << 1) + 1
      if (c > end) {
        break
      }
      c += +(c < end && _keys[c] < _keys[c + 1])
      if (_keys[c] <= key) {
        break
      }
      _keys[p] = _keys[c]
      _vals[p] = _vals[c]
      p = c
    }

    _keys[p] = key
    _vals[p] = val
  }

  /**
   * Sorts the entries of this heap by their keys in ascending
   * order. This method destroys the heap property and should
   * only be called after all elements have been added.
   */
  sort(): void {
    const { _keys, _vals, _pos } = this
    assert(0 === _pos, 'CappedMaxHeap().sort(): Heap is not full yet.')

    const swap = (i: number, j: number) => {
      let _key = _keys[i]
      _keys[i] = _keys[j]
      _keys[j] = _key
      let _val = _vals[i]
      _vals[i] = _vals[j]
      _vals[j] = _val
    }

    const sort2 = (from: number) => {
      if (_keys[from] > _keys[from + 1]) {
        swap(from, from + 1)
      }
    }

    const sort3 = (from: number) => {
      sort2(from)
      if (_keys[from + 1] > _keys[from + 2]) {
        swap(from + 1, from + 2)
        sort2(from)
      }
    }

    // Quick sort with median-of-3 pivotization
    const sort = (from: number, until: number) => {
      switch (until - from) {
        case 0:
          return
        case 1:
          return
        case 2:
          return sort2(from)
        case 3:
          return sort3(from)
      }

      // use median of 3 as pivot
      const mid = (from + until) >>> 1
      sort3(mid - 1)
      const piv = _keys[mid]
      swap(from, mid)

      // tri-partition such that:
      //   * [from,l) contains elements less than the pivot
      //   * [l,r) contains elements equal to the pivot
      //   * [r,until) contains elements greater than the pivot
      let l = from
      let r = from + 1
      for (let i = r; i < until; i++) {
        let ki = _keys[i]
        if (ki <= piv) {
          swap(i, r)
          if (ki < piv) {
            swap(l++, r)
          }
          r++
        }
      }

      sort(from, l)
      sort(r, until)
    }

    sort(0, _keys.length)
  }
}
