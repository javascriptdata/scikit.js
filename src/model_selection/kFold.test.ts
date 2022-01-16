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
import { KFold } from './kFold'
import { alea } from 'seedrandom'
import '../jestTensorMatchers'
import { tf } from '../shared/globals'
type Tensor2D = tf.Tensor2D

describe('KFold', () => {
  const numRuns = 128

  const anyInput = () =>
    fc
      .tuple(fc.nat(512), fc.nat(8), fc.nat(8), fc.string())
      .map(([nSamples, nFeatures, nSplits, seed]) => {
        nSplits += 2
        nFeatures += 1
        nSamples += nSplits
        return [tf.rand([nSamples, nFeatures], alea(seed)), nSplits] as [
          Tensor2D,
          number
        ]
      })

  for (const shuffle of [false, true, undefined]) {
    it(`yields correct number of splits {shuffle: ${shuffle}}`, async () => {
      const testBody = async ([X, nSplits]: [Tensor2D, number]) => {
        tf.engine().startScope()
        try {
          const kFold = new KFold({ nSplits, shuffle })

          let nIter = 0
          for (const { trainIndex, testIndex } of kFold.split(X)) {
            expect(nIter++).toBeLessThan(nSplits)
            trainIndex.dispose()
            testIndex.dispose()
          }

          expect(nIter).toBe(nSplits)
        } finally {
          tf.engine().endScope()
        }
      }

      await fc.assert(fc.asyncProperty(anyInput(), testBody), { numRuns })
    })
  }

  for (const shuffle of [false, true, undefined]) {
    it(`yields int32 tensors {shuffle: ${shuffle}}`, async () => {
      const testBody = async ([X, nSplits]: [Tensor2D, number]) => {
        tf.engine().startScope()
        try {
          const kFold = new KFold({ nSplits, shuffle })

          for (const { trainIndex, testIndex } of kFold.split(X)) {
            expect(trainIndex.dtype).toBe('int32')
            expect(testIndex.dtype).toBe('int32')
          }
        } finally {
          tf.engine().endScope()
        }
      }

      await fc.assert(fc.asyncProperty(anyInput(), testBody), { numRuns })
    })
  }

  for (const shuffle of [false, true, undefined]) {
    it(`yields splits with reasonable test sizes {shuffle: ${shuffle}}`, async () => {
      const testBody = async ([X, nSplits]: [Tensor2D, number]) => {
        tf.engine().startScope()
        try {
          const kFold = new KFold({ nSplits, shuffle })

          const nSamples = X.shape[0]
          const chunk = Math.floor(nSamples / nSplits)

          for (const { testIndex } of kFold.split(X)) {
            expect(testIndex.shape[0]).toBeGreaterThanOrEqual(chunk)
            expect(testIndex.shape[0]).toBeLessThanOrEqual(chunk + 1)
          }
        } finally {
          tf.engine().endScope()
        }
      }

      await fc.assert(fc.asyncProperty(anyInput(), testBody), { numRuns })
    })
  }

  for (const shuffle of [false, undefined]) {
    it(`yields unshuffled indices in sequence {shuffle: ${shuffle}}`, async () => {
      const testBody = async ([X, nSplits]: [Tensor2D, number]) => {
        tf.engine().startScope()
        try {
          const kFold = new KFold({ nSplits, shuffle })

          for (const { trainIndex, testIndex } of kFold.split(X)) {
            const nTrain = trainIndex.shape[0]
            const nTest = testIndex.shape[0]
            expect(trainIndex.slice(0, nTrain - 1).add(1)).toBeAllLessOrClose(
              trainIndex.slice(1),
              {
                rtol: 0,
                atol: 0,
                broadcast: false,
                allowEmpty: true
              }
            )
            expect(testIndex.slice(0, nTest - 1).add(1)).toBeAllCloseTo(
              testIndex.slice(1),
              { rtol: 0, atol: 0, broadcast: false, allowEmpty: true }
            )
          }
        } finally {
          tf.engine().endScope()
        }
      }

      await fc.assert(fc.asyncProperty(anyInput(), testBody), { numRuns })
    })
  }

  for (const shuffle of [false, true, undefined]) {
    it(`yields each index the correct number of times {shuffle: ${shuffle}}`, async () => {
      const testBody = async ([X, nSplits]: [Tensor2D, number]) => {
        tf.engine().startScope()
        try {
          const kFold = new KFold({ nSplits, shuffle })

          const nSamples = X.shape[0]
          let trainCounter = tf.zeros([nSamples], 'int32')
          let testCounter = tf.zeros([nSamples], 'int32')

          for (const { trainIndex, testIndex } of kFold.split(X)) {
            const trainMask = tf.scatterND(
              trainIndex.reshape([-1, 1]),
              tf.ones(trainIndex.shape, 'int32'),
              [nSamples]
            )
            const testMask = tf.scatterND(
              testIndex.reshape([-1, 1]),
              tf.ones(testIndex.shape, 'int32'),
              [nSamples]
            )

            // make sure each sample appears exactly once in current split
            expect(tf.add(trainMask, testMask)).toBeAllCloseTo(
              tf.ones([nSamples]),
              { rtol: 0, atol: 0, broadcast: false }
            )

            // keep count
            trainCounter = trainCounter.add(trainMask)
            testCounter = testCounter.add(testMask)
          }

          // make sure every sample appears in test exactly once
          expect(testCounter).toBeAllCloseTo(tf.ones([nSamples], 'int32'), {
            rtol: 0,
            atol: 0,
            broadcast: false
          })
          // make sure every example eppears in train (nSplits - 1 times)
          expect(trainCounter).toBeAllCloseTo(
            tf.fill([nSamples], nSplits - 1, 'int32'),
            { rtol: 0, atol: 0, broadcast: false }
          )
        } finally {
          tf.engine().endScope()
        }
      }

      await fc.assert(fc.asyncProperty(anyInput(), testBody), { numRuns })
    })
  }
})
