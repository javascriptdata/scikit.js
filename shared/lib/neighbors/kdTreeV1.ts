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

import { Tensor2D } from '@tensorflow/tfjs'
import { Vec } from '../types'
import { assert } from '../typesUtils'
import { tf } from '../../globals'
import { Metric } from './metrics'
import { Neighborhood, NeighborhoodParams } from './neighborhood'
import * as randUtils from '../randUtils'
import { alea } from 'seedrandom'
import { NAryHeap } from './nAryHeap'

const MAX_LEAF_SIZE = 8

type Node = Branch | Leaf

class Branch {
  axis: number
  threshold: number
  child0: Node
  child1: Node

  constructor(axis: number, threshold: number, child0: Node, child1: Node) {
    this.axis = axis
    this.threshold = threshold
    this.child0 = child0
    this.child1 = child1
  }
}

class Leaf {
  indices: Int32Array
  entries: Tensor2D

  constructor(indices: Int32Array, entries: Tensor2D) {
    this.indices = indices
    this.entries = entries
  }
}

class HeapNode {
  nearest: Vec
  node: Node

  constructor(nearest: Vec, node: Node) {
    this.nearest = nearest
    this.node = node
  }
}

export class KdTreeV1 implements Neighborhood {
  private _size: number
  private _metric: Metric
  private _root: Node

  static async create({ metric, entries }: NeighborhoodParams)
  {
    const [M, N] = entries.shape
    const data = await entries.data()
    const idxs = new Int32Array(M)
    for (let i = 0; i < M; i++) {
      idxs[i] = i
    }

    const randInt = randUtils.randInt( alea(`KdTree[${M},${N}]`) )

    const swap = (i: number, j: number) => {
      const t = idxs[i]
      idxs[i] = idxs[j]
      idxs[j] = t
    }

    function mkTree( axis: number, from: number, until: number ): Node
    {
      if (until - from <= MAX_LEAF_SIZE) {
        const idx = idxs.subarray(from, until)
        return new Leaf(idx, entries.gather(idx))
      }

      let threshold: number
      const mid = from + until >>> 1

      // quick-select to split `points` along `axis`
      for (let pos = from, end = until;;)
      {
        threshold = data[ N * idxs[randInt(pos, end)] + axis ]
        let l = pos,
            r = pos
        for (let i = pos; i < end; i++)
        { let pi = data[ N * idxs[i] + axis ]
          if (pi <= threshold ) { swap(i, r)
          if (pi < threshold ) { swap(l++, r) } r++ }
        }
             if (l > mid) end = l
        else if (r < mid) pos = r
        else break
      }

      const ax = (axis + 1) % N
      return new Branch(
        axis,
        threshold,
        mkTree(ax, from, mid),
        mkTree(ax, mid, until)
      )
    }

    return new KdTreeV1(M, metric, mkTree(0, 0, M))
  }

  private constructor(size: number, metric: Metric, root: Node) {
    this._size = size
    this._metric = metric
    this._root = root
  }

  private *_nearest(
    k: number,
    queryPoint: Tensor2D,
    nearest: Float32Array,
    node: Node
  ): IterableIterator<[number, number]> {
    const { _metric } = this
    assert(queryPoint.shape[0] === 1, 'Only a single queryPoint allowed.')

    if (node instanceof Branch) {
      // BRANCH NODE
      // -----------
      let { axis, threshold, child0, child1 } = node

      const nax = nearest[axis]
      nearest[axis] = threshold
      const [distMin] = tf.tidy( () => _metric(queryPoint, tf.tensor2d(nearest, [1, nearest.length])).dataSync() )
      nearest[axis] = nax

      if (threshold < nax) {
        const tmp = child0
        child0 = child1
        child1 = tmp
      }

      const iter0 = this._nearest(k, queryPoint, nearest, child0)
      let val0: [number, number], done0: boolean | undefined

      for (;;) {
        ({ value: val0, done: done0 } = iter0.next())
        if (done0 || val0[0] > distMin) break
        yield val0
      }

      const nearer = nearest.slice()
      nearer[axis] = threshold

      const iter1 = this._nearest(k, queryPoint, nearer, child1)
      let { value: val1, done: done1 } = iter1.next()

      while (!done0 || !done1)
        if (done1 || !done0 && val0[0] <= val1[0]) {
          yield val0;
          ({ value: val0, done: done0 } = iter0.next())
        }
        else {
          yield val1;
          ({ value: val1, done: done1 } = iter1.next())
        }
    } else {
      // LEAF NODE
      // ---------
      const { indices, entries } = node

      const [dist, order] = tf.tidy(() => {
        const { values, indices } = tf.topk(_metric(queryPoint, entries).neg(), k, /*sorted=*/true)
        return [ values.dataSync(), indices.dataSync() ]
      })

      for (let i = 0; i < order.length; i++) {
        yield [-dist[i], indices[order[i]]]
      }
    }
  }

//  kNearest(
//    k: number,
//    queryPoints: Tensor2D
//  ): { distances: Tensor2D; indices: Tensor2D }
//  {
//    const [M, N] = queryPoints.shape
//
//    return tf.tidy( () => {
//      if (1 < M) {
//        const dist: Tensor2D[] = []
//        const idxs: Tensor2D[] = []
//
//        for (const qp of queryPoints.unstack()) {
//          const { distances, indices } = this.kNearest(k, qp.reshape([1, N]))
//          dist.push(distances)
//          idxs.push(indices)
//        }
//
//        return {
//          distances: tf.concat(dist),
//          indices: tf.concat(idxs)
//        }
//      }
//
//      const dist = new Float32Array(k)
//      const idxs = new Int32Array(k)
//
//      let j = 0
//      for (const [d, i] of this._nearest(
//        k,
//        queryPoints,
//        queryPoints.dataSync() as Float32Array,
//        this._root
//      )) {
//        dist[j] = d
//        idxs[j] = i
//        if (++j >= k) {
//          break
//        }
//      }
//
//      return {
//        distances: tf.tensor2d(dist.subarray(0, j), [1, j], 'float32'),
//        indices: tf.tensor2d(idxs.subarray(0, j), [1, j], 'int32')
//      }
//    })
//  }

  kNearest(
    k: number,
    queryPoints: Tensor2D
  ): { distances: Tensor2D; indices: Tensor2D }
  {
    const { _metric, _root, _size } = this
    const [M, N] = queryPoints.shape
    k = Math.min(k, _size)

    return tf.tidy( () => {
      if (1 < M) {
        const dists: Tensor2D[] = []
        const indxs: Tensor2D[] = []

        for (const qp of queryPoints.unstack()) {
          const { distances, indices } = this.kNearest(k, qp.reshape([1, N]))
          dists.push(distances)
          indxs.push(indices)
        }

        return {
          distances: tf.concat(dists),
          indices: tf.concat(indxs)
        }
      }

      const dists = new Float32Array(k)
      const indxs = new Int32Array(k)

      const heap = new NAryHeap<HeapNode | number>()
      heap.add( 0, new HeapNode(queryPoints.dataSync(), _root) )

      for (let i = 0; i < k;)
      {
        let dist = heap.minKey
        let item = heap.minVal
        heap.popMin()

        if (item instanceof HeapNode) {
          let { nearest, node } = item

          while (node instanceof Branch)
          {
            let { axis, threshold, child0, child1 } = node

            const nearer = nearest.slice()
            nearer[axis] = threshold
            const [key] = tf.tidy(() => _metric(queryPoints, tf.tensor(nearer, [1, N])).dataSync())

            if (threshold < nearest[axis]) {
              node = child1
              child1 = child0
            }
            else {
              node = child0
            }

            heap.add(key, new HeapNode(nearer, child1))
          }

          const { entries, indices } = node
          const dist = tf.tidy(() => _metric(entries, queryPoints).dataSync())

          for (let i = 0; i < indices.length; i++) {
            heap.add(dist[i], indices[i])
          }
          continue
        }

        dists[i] = dist
        indxs[i] = item
        i++
      }

      return {
        distances: tf.tensor(dists, [1, k], 'float32'),
        indices: tf.tensor(indxs, [1, k], 'int32')
      }
    })
  }
}
