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

const child = (parent: number) => (parent << 1) + 1
const parent = (child: number) => child - 1 >> 1

const ceilPow2 = (int32: number) => {
  assert(
    0 <= int32 && int32 <= 0x4000_0000,
    'ceilPow2(x): x must be in range [0, 1<<30].'
  )
  int32 = Math.ceil(int32)
  int32 = Math.max(int32, 1)
  return 0x8000_0000 >>> (-1 + Math.clz32(int32 - 1))
}

type Vec = {
  [i: number]: number
  readonly length: number
  slice(start?: number, end?: number): Vec
  subarray(start?: number, end?: number): Vec
}

export class KdTreeV4 implements Neighborhood {
  private _nSamples: number
  private _nFeatures: number

  private _metric: TreeMetric
  private _points: Vec[]

  private _bBoxes: Float32Array[]
  private _offsets: Int32Array
  private _indices: Int32Array

  private constructor(nSamples: number, nFeatures: number, metric: TreeMetric, points: Vec[], bBoxes: Float32Array[], offsets: Int32Array, indices: Int32Array) {
    this._nSamples = nSamples
    this._nFeatures = nFeatures

    this._metric = metric
    this._points = points

    this._bBoxes = bBoxes
    this._offsets = offsets
    this._indices = indices
    Object.freeze(this)
  }

  static async create({ metric: { treeMetric }, entries }: NeighborhoodParams)
  {
    const [nSamples, nFeatures] = entries.shape

    const data = await entries.data()

    const points: Vec[] = []
    for (let i = 0; i < nSamples;) {
      points.push( data.subarray(nFeatures * i, nFeatures * ++i) )
    }

    const indices = new Int32Array(nSamples)
    for (let i = 0; i < nSamples; i++) {
      indices[i] = i
    }

    const nLeafs = ceilPow2(nSamples / MAX_LEAF_SIZE)
    const nNodes = nLeafs * 2 - 1

    const leaf0 = nNodes - nLeafs

    const offsets = new Int32Array(nLeafs + 1)
    const bBoxes = function() {
      const n = nFeatures * 2
      const flat = new Float32Array(nNodes * n)
      const bBoxes: Float32Array[] = []
      for (let i = 0; i < nNodes; ) {
        bBoxes.push( flat.subarray(n * i, n * ++i) )
      }
      return bBoxes
    }()

    const randInt = randUtils.randInt( alea(`KdTree[${nSamples},${nFeatures}]`) )

    const swapIndices = (i: number, j: number) => {
      const t = indices[i]
      indices[i] = indices[j]
      indices[j] = t
    }

    const buildTree = ( node: number, axis: number, from: number, until: number ) =>
    {
      // LEAF CASE
      // ---------
      if (leaf0 <= node) {
        const leaf = node - leaf0
        offsets[leaf] = from
        offsets[leaf + 1] = until

        const bBox = bBoxes[node]
        for (let i = 0; i < bBox.length; i++) {
          bBox[i] = i % 2 ? -Infinity : +Infinity
        }

        for (let i = from; i < until; i++) {
          const j = indices[i]
          for (let k = 0; k < bBox.length;) {
            const djk = data[nFeatures * j + (k >>> 1)]
            bBox[k] = Math.min(bBox[k++], djk)
            bBox[k] = Math.max(bBox[k++], djk)
          }
        }

        return
      }

      // BRANCH CASE
      // -----------
      const mid = from + until >>> 1

      // quick-select to split `points` along `axis` in half
      for (let pos = from, end = until;;)
      {
        const threshold = data[ nFeatures * indices[randInt(pos, end)] + axis ]
        let l = pos,
            r = pos
        for (let i = pos; i < end; i++)
        { let pi = data[ nFeatures * indices[i] + axis ]
          if (pi <= threshold ) { swapIndices(i, r)
          if (pi < threshold ) { swapIndices(l++, r) } r++ }
        }
             if (l > mid) end = l
        else if (r < mid) pos = r
        else break
      }

      const c = child(node)
      axis = (axis + 1) % nFeatures
      buildTree(c, axis, from, mid)
      buildTree(c + 1, axis, mid, until)

      const aBox = bBoxes[c]
      const bBox = bBoxes[c + 1]
      const cBox = bBoxes[node]
      for (let j = 0; j < cBox.length;) {
        cBox[j] = Math.min(aBox[j], bBox[j++])
        cBox[j] = Math.max(aBox[j], bBox[j++])
      }
    }

    buildTree(0, 0, 0, nSamples)

    const swapData = (i: number, j: number) => {
      const ni = nFeatures * i
      const nj = nFeatures * j
      for (let k = 0; k < nFeatures; k++) {
        const ik = ni + k
        const jk = nj + k
        const di = data[ik]
        data[ik] = data[jk]
        data[jk] = di
      }
    }

    // apply permutations (given by indices) to data
    for (let perm = indices.slice(), i = 0; i < nSamples; i++)
    {
      // permutation cycle
      for (let j = i;;)
      {
        let k = perm[j]
        perm[j] = j
        if (k === i) {
          break
        }
        swapData(j, j = k)
      }
    }

    return new KdTreeV4(
      nSamples, nFeatures, treeMetric, points, bBoxes, offsets, indices
    )
  }

  kNearest(
    k: number,
    queryPoints: Tensor2D
  ): { distances: Tensor2D; indices: Tensor2D }
  {
    const { _nSamples, _nFeatures, _metric, _points, _bBoxes, _offsets, _indices } = this
    k = Math.min(k, _nSamples)

    const [nQueries, nDim] = queryPoints.shape
    assert(
      _nFeatures === nDim,
      'KNeighbors: X_train.shape[1] must equal X_predict.shape[1].'
    )

    const dists = new Float32Array(nQueries * k)
    const indxs = new Int32Array(nQueries * k)

    const leaf0 = parent(_bBoxes.length - 1) + 1

    if (0 < k && 0 < nQueries) {
      const query = queryPoints.dataSync() as Vec

      let heap: CappedMaxHeap
      let queryPt: Vec

      const knn = ( node: number, minDist: number ) =>
      {
        if (minDist >= heap.maxKey) {
          return
        }
        if (node < leaf0) {
          const c = child(node)
          const dist0 = _metric.distToBBox(queryPt, _bBoxes[c])
          const dist1 = _metric.distToBBox(queryPt, _bBoxes[c + 1])
          if (dist0 <= dist1) {
            knn(c, dist0)
            knn(c + 1, dist1)
          }
          else {
            knn(c + 1, dist1)
            knn(c, dist0)
          }
        }
        else {
          node -= leaf0
          const from = _offsets[node]
          const until = _offsets[node + 1]
          for (let i = from; i < until; i++) {
            const dist = _metric(queryPt, _points[i])
            heap.add(dist, _indices[i])
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
        knn(0, _metric.distToBBox(queryPt, _bBoxes[0]))
      }
    }

    return {
      distances: tf.tensor(dists, [nQueries, k], 'float32'),
      indices: tf.tensor(indxs, [nQueries, k], 'int32')
    }
  }

//  kNearest(
//    k: number,
//    queryPoints: Tensor2D
//  ): { distances: Tensor2D; indices: Tensor2D }
//  {
//    const { _nSamples, _nFeatures, _metric, _points, _bBoxes, _offsets, _indices } = this
//    k = Math.min(k, _nSamples)
//
//    const [nQueries, nDim] = queryPoints.shape
//    assert(
//      _nFeatures === nDim,
//      'KNeighbors: X_train.shape[1] must equal X_predict.shape[1].'
//    )
//
//    const dists = new Float32Array(nQueries * k)
//    const indxs = new Int32Array(nQueries * k)
//
//    const leaf0 = parent(_bBoxes.length - 1) + 1
//
//    if (0 < k && 0 < nQueries) {
//      const query = queryPoints.dataSync() as Vec
//
//      let heap: CappedMaxHeap
//      let queryPt: Vec
//
//      const stackNode = new Int32Array(32)
//      const stackDist = new Float32Array(32)
//
//      for (let q = 0; q < nQueries; q++) {
//        queryPt = query.subarray( nDim * q, nDim * (q + 1) )
//        const off = k * q
//        const end = k + off
//        heap = new CappedMaxHeap(
//          dists.subarray(off, end),
//          indxs.subarray(off, end)
//        )
//
//        stackDist[0] = _metric.distToBBox(queryPt, _bBoxes[0])
//        stackNode[0] = 0
//        let depth = 1
//
//        do {
//          let minDist = stackDist[--depth]
//          if (minDist >= heap.maxKey) {
//            continue
//          }
//
//          let node = stackNode[depth]
//
//          while (node < leaf0) {
//            const c = child(node)
//            const dist0 = _metric.distToBBox(queryPt, _bBoxes[c])
//            const dist1 = _metric.distToBBox(queryPt, _bBoxes[c + 1])
//            if (dist0 <= dist1) {
//              minDist = dist0
//              node = c
//              stackDist[depth] = dist1
//              stackNode[depth] = c + 1
//            }
//            else {
//              minDist = dist1
//              node = c + 1
//              stackDist[depth] = dist0
//              stackNode[depth] = c
//            }
//            ++depth
//          }
//
//          node -= leaf0
//          const from = _offsets[node]
//          const until = _offsets[node + 1]
//          for (let i = from; i < until; i++) {
//            const dist = _metric(queryPt, _points[i])
//            heap.add(dist, _indices[i])
//          }
//        }
//        while (0 <= depth)
//      }
//    }
//
//    return {
//      distances: tf.tensor(dists, [nQueries, k], 'float32'),
//      indices: tf.tensor(indxs, [nQueries, k], 'int32')
//    }
//  }
}
