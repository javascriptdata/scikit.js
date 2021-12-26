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
import { assert } from '../typesUtils'
import { tf } from '../../globals'
import { TreeMetric } from './metrics'
import { Neighborhood, NeighborhoodParams } from './neighborhood'
import * as randUtils from '../randUtils'
import { alea } from 'seedrandom'
import { NAryHeap } from './nAryHeap'
import { CappedMaxHeap } from './cappedMaxHeap'

const MAX_LEAF_SIZE = 8

type Vec = Float32Array
type Node = Branch | Int32Array

class Branch {
  axis: number
  threshold: number
  bBox: Vec
  child0: Node
  child1: Node

  constructor(axis: number, threshold: number, bBox: Vec, child0: Node, child1: Node) {
    this.axis = axis
    this.threshold = threshold
    this.bBox = bBox
    this.child0 = child0
    this.child1 = child1
  }
}

export class KdTreeV2 implements Neighborhood {
  private _nSamples: number
  private _nFeatures: number
  private _data: Vec
  private _metric: TreeMetric
  private _root: Node

  static async create({ metric: { treeMetric }, entries }: NeighborhoodParams)
  {
    const [nSamples, nFeatures] = entries.shape
    const data = await entries.data() as Vec
    const idxs = new Int32Array(nSamples)
    for (let i = 0; i < nSamples; i++) {
      idxs[i] = i
    }

    const randInt = randUtils.randInt( alea(`KdTree[${nSamples},${nFeatures}]`) )

    const swap = (i: number, j: number) => {
      const t = idxs[i]
      idxs[i] = idxs[j]
      idxs[j] = t
    }

    const bBox = (child0: Node, child1: Node) => {
      const result = new Float32Array(nFeatures * 2)
      for (let i = 0; i < result.length; i++) {
        result[i] = i % 2 ? -Infinity : +Infinity
      }

      for (const child of [child0, child1])
      {
        if (child instanceof Branch) {
          const { bBox } = child
          for (let j = 0; j < result.length;) {
            result[j] = Math.min(result[j], bBox[j++])
            result[j] = Math.max(result[j], bBox[j++])
          }
        }
        else {
          // (leaf) child instanceof Int32Array
          for (const i of child) {
            for (let j = 0; j < result.length;) {
              const dij = data[nFeatures * i + (j >>> 1)]
              result[j] = Math.min(result[j++], dij)
              result[j] = Math.max(result[j++], dij)
            }
          }
        }
      }

      return result
    }

    function mkTree( axis: number, from: number, until: number ): Node
    {
      if (until - from <= MAX_LEAF_SIZE) {
        return idxs.subarray(from, until)
      }

      let threshold: number
      const mid = from + until >>> 1

      // quick-select to split `points` along `axis`
      for (let pos = from, end = until;;)
      {
        threshold = data[ nFeatures * idxs[randInt(pos, end)] + axis ]
        let l = pos,
            r = pos
        for (let i = pos; i < end; i++)
        { let pi = data[ nFeatures * idxs[i] + axis ]
          if (pi <= threshold ) { swap(i, r)
          if (pi < threshold ) { swap(l++, r) } r++ }
        }
             if (l > mid) end = l
        else if (r < mid) pos = r
        else break
      }

      const ax = (axis + 1) % nFeatures
      const child0 = mkTree(ax, from, mid)
      const child1 = mkTree(ax, mid, until)
      return new Branch(
        axis,
        threshold,
        bBox(child0, child1),
        child0, child1
      )
    }

    return new KdTreeV2(nSamples, nFeatures, data, treeMetric, mkTree(0, 0, nSamples))
  }

  private constructor(nSamples: number, nFeatures: number, data: Vec, metric: TreeMetric, root: Node) {
    this._nSamples = nSamples
    this._nFeatures = nFeatures
    this._data = data
    this._metric = metric
    this._root = root
  }

//  private _kNearest( queryPoint: Vec, dists: Vec, indxs: Int32Array )
//  {
//    const { _nFeatures, _data, _metric, _root } = this
//
//    const heap = new NAryHeap<Branch | number>()
//    const enqueue = (node: Node) => {
//      if (node instanceof Branch) {
//        const minDist = _metric.distToBBox(queryPoint, node.bBox)
//        heap.add(minDist, node)
//      }
//      else {
//        for (const i of node) {
//          const dist = _metric(queryPoint, _data.subarray(_nFeatures * i, _nFeatures * (i + 1)))
//          heap.add(dist, i)
//        }
//      }
//    }
//    enqueue(_root)
//
//    for (let i = 0; i < dists.length;)
//    {
//      const key = heap.minKey
//      let val: Node | number = heap.minVal
//      heap.popMin()
//      if (val instanceof Branch)
//      {
//        do {
//          let { axis, threshold, child0, child1 } = val as Branch
//          if (threshold < queryPoint[axis]) {
//            val = child1
//            child1 = child0
//          }
//          else {
//            val = child0
//          }
//          enqueue(child1)
//        }
//        while (val instanceof Branch)
//        enqueue(val)
//        continue
//      }
//      dists[i] = key
//      indxs[i] = val
//      i++
//    }
//  }

  private _kNearest( queryPoint: Vec, dists: Vec, indxs: Int32Array )
  {
    const { _nFeatures, _data, _metric, _root } = this

    // min heap of nodes sorted by positive distance to bounding box
    const nodes = new NAryHeap<Branch>()
    // min heap of leaves sorted by negative distance
    const leafs = new CappedMaxHeap(dists, indxs)

    const enqueue = (node: Node) => {
      if (node instanceof Branch) {
        const minDist = _metric.distToBBox(queryPoint, node.bBox)
        nodes.add(minDist, node)
      }
      else {
        for (const i of node) {
          const dist = _metric(queryPoint, _data.subarray(_nFeatures * i, _nFeatures * (i + 1)))
          leafs.add(dist, i)
        }
      }
    }
    enqueue(_root)

    while ( !(leafs.maxKey <= nodes.minKey) )
    {
      let node: Node = nodes.popMin()
      do {
        let { axis, threshold, child0, child1 } = node as Branch
        if (threshold < queryPoint[axis]) {
          node = child1
          child1 = child0
        }
        else {
          node = child0
        }
        enqueue(child1)
      }
      while (node instanceof Branch)
      enqueue(node)
    }

    leafs.sort()
  }

  kNearest(
    k: number,
    queryPoints: Tensor2D
  ): { distances: Tensor2D; indices: Tensor2D }
  {
    const { _nSamples, _nFeatures } = this
    k = Math.min(k, _nSamples)

    const [nQueries, nDim] = queryPoints.shape
    assert(
      _nFeatures === nDim,
      'KNeighbors: X_train.shape[1] must equal X_predict.shape[1].'
    )

    const query = queryPoints.dataSync() as Vec
    const dists = new Float32Array(nQueries * k)
    const indxs = new Int32Array(nQueries * k)

    if (0 < k) {
      for (let q = 0; q < nQueries; q++) {
        this._kNearest(
          query.subarray( nDim * q, nDim * (q + 1) ),
          dists.subarray( k * q, k * (q + 1) ),
          indxs.subarray( k * q, k * (q + 1) )
        )
      }
    }

    return {
      distances: tf.tensor(dists, [nQueries, k], 'float32'),
      indices: tf.tensor(indxs, [nQueries, k], 'int32')
    }
  }

//  private *_nearest( queryPoint: Vec, node: Node ): IterableIterator<[number, number]>
//  {
//    const { _nFeatures, _data, _metric } = this
//    if (node instanceof Branch) {
//      let { axis, threshold, child0, child1 } = node
//
//      if (threshold < queryPoint[axis]) {
//        let c0 = child0
//        child0 = child1
//        child1 = c0
//      }
//
//      const distMin = child1 instanceof Branch
//        ? _metric.distToBBox(queryPoint, child1.bBox)
//        : 0
//
//      const iter0 = this._nearest(queryPoint, child0)
//      let val0: [number, number], done0: boolean | undefined
//
//      for (;;) {
//        ({ value: val0, done: done0 } = iter0.next())
//        if (done0 || val0[0] > distMin) break
//        yield val0
//      }
//
//      const iter1 = this._nearest(queryPoint, child1)
//      let { value: val1, done: done1 } = iter1.next()
//
//      while (!done0 || !done1) {
//        if (done1 || !done0 && val0[0] <= val1[0]) {
//          yield val0;
//          ({ value: val0, done: done0 } = iter0.next())
//        }
//        else {
//          yield val1;
//          ({ value: val1, done: done1 } = iter1.next())
//        }
//      }
//    }
//    else {
//      const dist = Float32Array.from(
//        node,
//        (i) => _metric(
//          queryPoint,
//          _data.subarray(_nFeatures * i, _nFeatures * (i + 1))
//        )
//      )
//
//      const order = node.map((_, i) => i)
//      order.sort((i, j) => dist[i] - dist[j])
//
//      for (const i of order) {
//        yield [dist[i], node[i]]
//      }
//    }
//  }
//
//  kNearest(
//    k: number,
//    queryPoints: Tensor2D
//  ): { distances: Tensor2D; indices: Tensor2D }
//  {
//    const { _nSamples, _nFeatures, _root } = this
//    k = Math.min(k, _nSamples)
//
//    const [nQueries, nDim] = queryPoints.shape
//    assert(
//      _nFeatures === nDim,
//      'KNeighbors: X_train.shape[1] must equal X_predict.shape[1].'
//    )
//
//    const query = queryPoints.dataSync() as Vec
//    const dists = new Float32Array(nQueries * k)
//    const indxs = new Int32Array(nQueries * k)
//
//    if (k > 0) {
//      for (let q = 0; q < nQueries; q++) {
//        const iter = this._nearest(
//          query.subarray( nDim * q, nDim * (q + 1) ),
//          _root
//        )
//        let n = 0
//        for (const [d, i] of iter) {
//          dists[ k * q + n ] = d
//          indxs[ k * q + n ] = i
//          if (++n >= k) {
//            break
//          }
//        }
//      }
//    }
//
//    return {
//      distances: tf.tensor(dists, [nQueries, k], 'float32'),
//      indices: tf.tensor(indxs, [nQueries, k], 'int32')
//    }
//  }
}
