/**
*  @license
* Copyright 2022, JsData. All rights reserved.
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
import { tf } from '../shared/globals'
import { alea } from 'seedrandom'
import { Neighborhood, NeighborhoodParams } from './Neighborhood'
import { lhs, shuffle } from '../randUtils'
import { minkowskiMetric } from './Metric'
import { polyfillUnique } from '../tfUtils'
import '../jestTensorMatchers'

export function neighborhoodGenericTests(
  name: string,
  buildNeighborhood: (params: NeighborhoodParams) => Promise<Neighborhood>
) {
  describe(`${name} [generic tests]`, () => {
    for (const p of [1, 2, Infinity]) {
      const numRuns = 128

      const metric = minkowskiMetric(p)

      const anyFloat = () => fc.double(-(2 ** 16), +(2 ** 16))

      it(`kNearest(1, ...) returns distinct points as closest to themselves { metric: ${metric.name} }`, async () => {
        const anyDistinctPoints = () =>
          fc
            .tuple(fc.nat(256), fc.nat(4), fc.string(), anyFloat(), anyFloat())
            .chain(([nSamples, nDim, seed, scale, offset]) => {
              ++nSamples
              ++nDim

              const rng = alea(seed)

              const entries = lhs(rng)(nSamples, nDim).map((row) =>
                Array.from(row, (x) => (x - 0.5) * scale + offset)
              )

              const idx = new Int32Array(nSamples)
              for (let i = 0; i < nSamples; i++) {
                idx[i] = i
              }
              shuffle(rng)(idx)

              return fc
                .nat(nSamples)
                .map<[tf.Tensor2D, tf.Tensor1D]>((nQueries) => [
                  tf.tensor2d(entries),
                  tf.tensor1d(idx.subarray(0, ++nQueries))
                ])
            })

        const testBody = async ([entries, queryIdx]: [
          tf.Tensor2D,
          tf.Tensor1D
        ]) => {
          tf.engine().startScope()
          try {
            const queries = entries.gather(queryIdx)

            const neighborhood = await buildNeighborhood({ entries, metric })

            const { distances, indices } = neighborhood.kNearest(1, queries)

            const nQueries = queries.shape[0]
            expect(distances.abs().arraySync()).toEqual(
              new Array(nQueries).fill([0])
            )
            expect(indices.arraySync()).toEqual(
              queryIdx.reshape([nQueries, 1]).arraySync()
            )
          } finally {
            tf.engine().endScope()
            entries.dispose()
            queryIdx.dispose()
          }
        }

        tf.engine().startScope()
        try {
          await fc.assert(fc.asyncProperty(anyDistinctPoints(), testBody), {
            numRuns
          })
        } finally {
          tf.engine().endScope()
        }
      })

      it(`kNearest(k, ...) returns nearest k points { metric: ${metric.name} }`, async () => {
        const anyPoints = (nSamples: number, ndim: number) =>
          fc
            .array(
              fc.array(anyFloat(), { minLength: ndim, maxLength: ndim }),
              { minLength: nSamples, maxLength: nSamples }
            )
            .map<tf.Tensor2D>(tf.tensor)

        const anyInput = () =>
          fc
            .tuple(fc.nat(256), fc.nat(8), fc.nat(4))
            .chain(([nSamples, nQueries, nDim]) => {
              ++nSamples
              ++nQueries
              ++nDim
              return fc.tuple(
                anyPoints(nSamples, nDim),
                anyPoints(nQueries, nDim),
                fc.nat(nSamples - 1).map((k) => ++k)
              )
            })

        polyfillUnique(tf)

        const testBody = async ([entries, queries, k]: [
          tf.Tensor2D,
          tf.Tensor2D,
          number
        ]) => {
          tf.engine().startScope()
          try {
            const neighborhood = await buildNeighborhood({ entries, metric })

            const nSamples = entries.shape[0]
            const nQueries = queries.shape[0]

            const { distances, indices } = neighborhood.kNearest(k, queries)

            expect(distances.shape).toEqual([nQueries, k])
            expect(indices.shape).toEqual([nQueries, k])

            const dists = distances.unstack()
            const indxs = indices.unstack()
            const queryPts = queries.unstack()

            for (let i = 0; i < nQueries; i++) {
              tf.engine().startScope()
              try {
                const allDist = metric.tensorDistance(entries, queryPts[i])

                // make sure distances match indices
                expect(dists[i]).toBeAllCloseTo(allDist.gather(indxs[i]), {
                  broadcast: false
                })

                // make sure no duplicate indices are returned
                expect(tf.unique(indxs[i]).values).toBeAllCloseTo(indxs[i], {
                  rtol: 0,
                  atol: 0,
                  broadcast: false
                })

                const mask = tf.scatterND(
                  indxs[i].reshape([-1, 1]),
                  tf.ones([k], 'bool'),
                  [nSamples]
                )
                const maxDist = dists[i].max()

                expect(allDist).toBeAllGreaterOrClose(
                  tf.where(mask, 0, maxDist),
                  { broadcast: false }
                )
              } finally {
                tf.engine().endScope()
              }
            }
          } finally {
            tf.engine().endScope()
          }
        }

        tf.engine().startScope()
        try {
          await fc.assert(fc.asyncProperty(anyInput(), testBody), { numRuns })
        } finally {
          tf.engine().endScope()
        }
      })
    }
  })
}
