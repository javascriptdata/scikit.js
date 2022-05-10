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

import './jestTensorMatchers'
import * as tf from '@tensorflow/tfjs'
import { setBackend } from './index'
setBackend(tf)

describe('Custom Jest Tensor Matchers', () => {
  it('passes handcrafted tests', () => {
    expect([1, 1, 1]).toBeAllCloseTo(1, { rtol: 0, atol: 0 })

    expect([1, 2, 3, 4]).toBeAllLessOrClose(4, { rtol: 0, atol: 0 })

    expect([
      [5, 6],
      [7, 8]
    ]).toBeAllGreaterOrClose(5, { rtol: 0, atol: 0 })

    expect([
      [30, 3],
      [10, 1],
      [20, 2]
    ]).toBeAllGreaterOrClose([10, 1], { rtol: 0, atol: 0 })

    expect([
      [30, 3],
      [1, 10],
      [20, 2]
    ]).toBeAllGreaterOrClose([[3], [1], [2]], { rtol: 0, atol: 0 })

    expect([
      [
        [+1.01, -2.02, +3.03],
        [-4.04, +5.05, -6.06]
      ],
      [
        [-10.1, +20.2, +30.3],
        [+40.4, -50.5, -60.6]
      ]
    ]).toBeAllCloseTo(
      [
        [
          [+1, -2, +3],
          [-4, +5, -6]
        ],
        [
          [-10, +20, +30],
          [+40, -50, -60]
        ]
      ],
      { rtol: 0.0101, atol: 0 }
    )

    expect([
      [
        [+1, +20, -3, -40],
        [-50, -6, +70, -8]
      ]
    ]).toBeAllCloseTo(
      [
        [
          [+1.01, +20.01, -3.01, -40.01],
          [-49.99, -5.99, +69.99, -7.99]
        ],
        [
          [+1.01, +19.99, -2.99, -40.01],
          [-50.01, -5.99, +69.99, -8.01]
        ]
      ],
      { rtol: 0, atol: 0.0101 }
    )

    expect([0.99, 1.99]).toBeAllLessNotClose([1, 2], { rtol: 0, atol: 0.0095 })

    expect([1, 2]).not.toBeAllCloseTo([1.1, 2.1], { rtol: 0.05 })

    expect(() =>
      expect([1, 2, 3, 4]).toBeAllCloseTo([1, 2.1, 3, 4], { rtol: 0.01 })
    ).toThrow()

    expect(() => expect([]).toBeAllCloseTo([])).toThrow()
    expect(() => expect([1]).toBeAllCloseTo([])).toThrow()
    expect(() => expect([]).toBeAllCloseTo([2])).toThrow()

    expect([]).toBeAllCloseTo([], { allowEmpty: true })
    expect([1]).toBeAllCloseTo([], { allowEmpty: true })
    expect([]).toBeAllCloseTo([2], { allowEmpty: true })
  })
})
