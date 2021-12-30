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
import { CappedMaxHeap } from './cappedMaxHeap'

const MAX_LEAF_SIZE = 16

type Vec = Float32Array
type Node = Branch | Leaf

class Branch {
  bBox: Vec
  child0: Node
  child1: Node

  constructor(child0: Node, child1: Node) {
    const bBox0 = child0.bBox.slice()
    const bBox1 = child1.bBox

    for (let j = 0; j < bBox0.length;) {
      bBox0[j] = Math.min(bBox0[j], bBox1[j++])
      bBox0[j] = Math.max(bBox0[j], bBox1[j++])
    }

    this.bBox = bBox0
    this.child0 = child0
    this.child1 = child1
  }
}

class Leaf {
  bBox: Vec
  indices: Int32Array

  constructor( bBox: Vec, indices: Int32Array ) {
    this.bBox = bBox
    this.indices = indices
  }
}

export class KdTreeV3 implements Neighborhood {
  private _nSamples: number
  private _nFeatures: number
  private _data: Vec[]
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

    function mkTree( axis: number, from: number, until: number ): Node
    {
      if (until - from <= MAX_LEAF_SIZE) {
        const bBox = new Float32Array(nFeatures << 1)
        for (let i = 0; i < bBox.length; i++) {
          bBox[i] = i % 2 ? -Infinity : +Infinity
        }
        const idx = idxs.subarray(from, until)
        for (const i of idx) {
          for (let j = 0; j < bBox.length;) {
            const dij = data[nFeatures * i + (j >>> 1)]
            bBox[j] = Math.min(bBox[j++], dij)
            bBox[j] = Math.max(bBox[j++], dij)
          }
        }
        return new Leaf(bBox, idx)
      }

      const mid = from + until >>> 1

      // quick-select to split `points` along `axis` in half
      for (let pos = from, end = until;;)
      {
        const threshold = data[ nFeatures * idxs[randInt(pos, end)] + axis ]
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

      axis = (axis + 1) % nFeatures
      return new Branch(
        mkTree(axis, from, mid),
        mkTree(axis, mid, until)
      )
    }

    const points: Vec[] = []
    for (let i = 0; i < nSamples;) {
      points.push( data.subarray(nFeatures * i, nFeatures * ++i) )
    }

    return new KdTreeV3(nSamples, nFeatures, points, treeMetric, mkTree(0, 0, nSamples))
  }

  private constructor(nSamples: number, nFeatures: number, data: Vec[], metric: TreeMetric, root: Node) {
    this._nSamples = nSamples
    this._nFeatures = nFeatures
    this._data = data
    this._metric = metric
    this._root = root
  }

  kNearest(
    k: number,
    queryPoints: Tensor2D
  ): { distances: Tensor2D; indices: Tensor2D }
  {
    const { _nSamples, _nFeatures, _data, _metric, _root } = this
    k = Math.min(k, _nSamples)

    const [nQueries, nDim] = queryPoints.shape
    assert(
      _nFeatures === nDim,
      'KNeighbors: X_train.shape[1] must equal X_predict.shape[1].'
    )

    const dists = new Float32Array(nQueries * k)
    const indxs = new Int32Array(nQueries * k)

    if (0 < k && 0 < nQueries) {
      const query = queryPoints.dataSync() as Vec

      let heap: CappedMaxHeap
      let queryPt: Vec

      const knn = ( node: Node, minDist: number ) =>
      {
        if (minDist >= heap.maxKey) {
          return
        }
        if (node instanceof Branch) {
          const { child0, child1 } = node
          const dist0 = _metric.distToBBox(queryPt, child0.bBox)
          const dist1 = _metric.distToBBox(queryPt, child1.bBox)
          if (dist0 <= dist1) {
            knn(child0, dist0)
            knn(child1, dist1)
          }
          else {
            knn(child1, dist1)
            knn(child0, dist0)
          }
        }
        else {
          for (const i of node.indices) {
            const dist = _metric(queryPt, _data[i])
            heap.add(dist, i)
          }
        }
      }

      for (let q = 0; q < nQueries; q++) {
        queryPt = query.subarray( nDim * q, nDim * (q + 1) )
        const off = k * q
        const end = k + off
        heap = new CappedMaxHeap(
          dists.subarray(off, end),
          indxs.subarray(off, end)
        )
        knn(_root, _metric.distToBBox(queryPt, _root.bBox))
      }
    }

    return {
      distances: tf.tensor(dists, [nQueries, k], 'float32'),
      indices: tf.tensor(indxs, [nQueries, k], 'int32')
    }
  }
}
