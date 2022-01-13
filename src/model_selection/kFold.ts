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

import { tf } from '../shared/globals'
import { Tensor2D, Tensor1D } from '@tensorflow/tfjs-node'
import { assert } from '../typesUtils'
import { CrossValidator } from './crossValidator'
import { alea } from 'seedrandom'
import * as randUtils from '../randUtils'

export interface KFoldParams {
  /**
   * Number of ways in which the dataset is to be split.
   * Defaults to `5`. Larger numbers of splits will result
   * in increased computational cost, since the model is
   * trained `nSplits` time during cross validation. In
   * return, more training data is available in each split,
   * which allows utilize smaller datasets more efficiently.
   */
  nSplits?: number
  /**
   * If set to `true`, indices are shuffled before train/test
   * splitting. Defaults to `false`.
   */
  shuffle?: boolean
  /**
   * Random seed to be used for shuffling. Ignored if `nSplits = false`.
   */
  randomState?: number
}

/**
 * K-Fold cross-validator
 *
 * Generates train and test indices to split data in train/test subsets.
 * To generate these subsets, the dataset is split into k (about) evenly
 * sized chunks of consecutive elements. Each split takes another chunk
 * as test data and the remaining chunks are combined to be the training
 * data.
 *
 * Optionally, the indices can be shuffled before splitting it into chunks
 * (disabled by default).
 *
 * @example
 * ```js
 * import { KFold } from 'scikitjs'
 *
 * const kf = new KFold({ nSplits: 3 })
 *
 * const X = tf.range(0, 7).reshape([7, 1]) as Tensor2D
 *
 * console.log( 'nSplits:', kf.getNumSplits(X) )
 *
 * for (const { trainIndex, testIndex } of kf.split(X) )
 * {
 *   try {
 *     console.log( 'train:', trainIndex.toString() )
 *     console.log( 'test:',   testIndex.toString() )
 *   }
 *   finally {
 *     trainIndex.dispose()
 *      testIndex.dispose()
 *   }
 * }
 * ```
 */
export class KFold implements CrossValidator {
  nSplits: number
  shuffle: boolean
  randomState?: number

  constructor({
    nSplits = 5,
    shuffle = false,
    randomState
  }: KFoldParams = {}) {
    nSplits = Number(nSplits)
    assert(
      nSplits % 1 === 0 && nSplits > 1,
      'new KFold({nSplits}): nSplits must be an int greater than 1.'
    )
    this.nSplits = nSplits
    this.shuffle = !!shuffle
    this.randomState = randomState
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  public getNumSplits(X?: Tensor2D, y?: Tensor1D, groups?: Tensor1D): number {
    return this.nSplits
  }

  public *split(
    X: Tensor2D,
    y?: Tensor1D,
    groups?: Tensor1D
  ): IterableIterator<{ trainIndex: Tensor1D; testIndex: Tensor1D }> {
    const { nSplits, shuffle, randomState } = this

    const nSamples = X.shape[0]

    assert(
      nSplits <= nSamples,
      'KFold({nSplits})::split(X): nSplits must not be greater than X.shape[0].'
    )

    if (null != y) {
      assert(
        nSamples === y.shape[0],
        'KFold::split(X,y): X.shape[0] must equal y.shape[0].'
      )
    }

    if (null != groups) {
      assert(
        nSamples === groups.shape[0],
        'KFold::split(X,y,groups): X.shape[0] must equal groups.shape[0].'
      )
    }

    const range = new Int32Array(nSamples)
    for (let i = 0; i < range.length; i++) {
      range[i] = i
    }
    if (shuffle) {
      const rng = alea(randomState?.toString())
      randUtils.shuffle(rng)(range)
    }

    const chunkBase = Math.floor(nSamples / nSplits)
    let remainder = nSamples % nSplits

    for (let offset = 0; offset < nSamples; ) {
      const chunk = chunkBase + Number(remainder-- > 0)

      const train = new Int32Array(nSamples - chunk)
      train.set(range.subarray(0, offset), 0)
      train.set(range.subarray(offset + chunk), offset)

      const test = range.slice(offset, offset + chunk)

      yield {
        trainIndex: tf.tensor1d(train, 'int32'),
        testIndex: tf.tensor1d(test, 'int32')
      }

      offset += chunk
    }
  }
}
