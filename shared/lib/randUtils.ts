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

type int = number

export const randInt = (rand: () => number) => (from: int, until: int) => {
  if (from >= until)
    throw new Error('randInt(rng)(from,until): from must be less than until.')

  const result = (from + (until - from) * rand()) | 0
  return result
}

/**
 * Generates a random latin hypercube sampling, i.e. unique samples whose
 * features are evenly distributed over the range `[0,1)`. Latin hypercube
 * sampling is useful to create random non-duplicate samples that are
 * somewhat evenly distribute in the `[0,1)**nFeatures` hypercube.
 *
 * @see {@link https://en.wikipedia.org/wiki/Latin_hypercube_sampling}
 */
export const lhs = (rand: () => number) => (nSamples: int, nFeatures: int) => {
  const _randInt = randInt(rand)

  const result: Float64Array[] = []

  // init columns to linspace(0,1, nSamples)
  for (let i = nSamples; i-- > 0; ) {
    const row = new Float64Array(nFeatures)
    row.fill(i / (nSamples - 1 || 1))
    result.push(row)
  }

  // shuffle columns independently
  for (let n, i = nSamples; (n = i--) > 1; ) {
    const ri = result[i]
    for (let j = nFeatures; j-- > 0; ) {
      const k = _randInt(0, n),
        rij = ri[j]
      ri[j] = result[k][j]
      result[k][j] = rij
    }
  }

  return result
}

export const shuffle =
  (rand: () => number) =>
  <T>(array: { readonly length: number; [i: number]: T }) => {
    const _randInt = randInt(rand)
    for (let i = array.length; i > 1; ) {
      const j = _randInt(0, i--),
        tmp = array[i]
      array[i] = array[j]
      array[j] = tmp
    }
  }
