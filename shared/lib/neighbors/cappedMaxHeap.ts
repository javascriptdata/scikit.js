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

export class CappedMaxHeap {
  private _keys: Float32Array
  private _vals: Int32Array
  private _pos: number

  constructor( keys: Float32Array, vals: Int32Array ) {
    const len = keys.length
    assert(len > 0, 'new CappedMaxHeap(keys,vals): keys.length must be positive.')
    assert(len === vals.length, 'new CappedMaxHeap(keys,vals): keys.length must equal vals.length.')
    keys[0] = NaN
    this._keys = keys
    this._vals = vals
    this._pos = len
  }

  get maxKey() {
    return this._keys[0]
  }

  add( key: number, val: number ) {
    let { _keys, _vals, _pos: i } = this
    if (0 < i) {
      this._pos = --i
    }
    else if (key > _keys[0]) {
      return
    }
    const { length } = _keys

    // sift-down
    for (;;) {
      const p = i
      // fill parent gap with filler
      _keys[i] = key
      _vals[i] = val
      // c: leftmost child of i
      let c = (i << 1) + 1

      // find largest child that is larger than the filler
      const lastChild = Math.min(length, c + 2)
      for (;lastChild > c; c++)
        if (_keys[c] > _keys[i]) i = c

      // if parent is already the smallest value, stop
      if (i === p) break

      // move smallest value to root/parent
      _keys[p] = _keys[i]
      _vals[p] = _vals[i]
    }
  }

  sort()
  {
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
      if ( _keys[from + 1] > _keys[from + 2] ) {
        swap(from + 1, from + 2)
        sort2(from)
      }
    }

    // Quick sort with median-of-3 pivotization
    const sort = (from: number, until: number) => {
      switch (until - from) {
        case 0: return
        case 1: return
        case 2: return sort2(from)
        case 3: return sort3(from)
      }

      // use median of 3 as pivot
      const mid = from + until >>> 1
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
        { let ki = _keys[i]
          if (ki <= piv ) { swap(i, r)
          if (ki < piv ) { swap(l++, r) } r++ }
        }
      }

      sort(from, l)
      sort(r, until)
    }

    sort(0, _keys.length)
  }
}
