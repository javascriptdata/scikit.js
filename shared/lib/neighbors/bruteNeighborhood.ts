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

import { NAryHeap } from './nAryHeap'
import {
  Neighbor,
  Neighborhood,
  NeighborhoodEntry,
  NeighborhoodParams
} from './neighborhood'
import { Metric } from './metrics'

export class BruteNeighborhood<A, V> implements Neighborhood<A, V> {
  private _metric: Metric<A>
  private _entries: NeighborhoodEntry<A, V>[]

  constructor({ metric, entries = [] }: NeighborhoodParams<A, V>) {
    this._metric = metric
    this._entries = [
      ...(function* () {
        for (const { address, value } of entries) yield { address, value }
      })()
    ]
  }

  private *_sortedByDistance(
    addr: A,
    compareFn: (distance1: number, distance2: number) => -1 | 0 | 1
  ) {
    const { _metric, _entries } = this

    const heap = new NAryHeap<Neighbor<A, V>>(
      ({ distance: x }, { distance: y }) => compareFn(x, y)
    )

    for (const { address, value } of _entries) {
      const distance = _metric(addr, address)
      if (isNaN(distance) || distance < 0)
        throw new Error('KNeighborsBase: metric returned invalid distance.')
      heap.add({ address, value, distance })
    }

    for (let i = heap.size; i-- > 0; ) yield heap.popMin()
  }

  nearest(address: A) {
    return this._sortedByDistance(
      address,
      (x, y) => Math.sign(x - y) as -1 | 0 | 1
    )
  }

  farthest(address: A) {
    return this._sortedByDistance(
      address,
      (x, y) => Math.sign(y - x) as -1 | 0 | 1
    )
  }

  [Symbol.iterator](): IterableIterator<NeighborhoodEntry<A, V>> {
    return this._entries[Symbol.iterator]()
  }
}
