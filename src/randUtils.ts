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
export function _prng_restore(prng: any, xg: any, opts: any) {
  let state = opts && opts.state
  if (state) {
    if (typeof state == 'object') xg.copy(state, xg)
    prng.state = () => xg.copy(xg, {})
  }
}

export function alea(seed?: any, opts?: any) {
  let xg = new AleaGen(seed)

  let prng = () => xg.next()

  _prng_restore(prng, xg, opts)
  return prng
}

class AleaGen {
  c: number
  s0: number
  s1: number
  s2: number
  constructor(seed: any) {
    if (seed == null) seed = +new Date()

    let n = 0xefc8249d

    // Apply the seeding algorithm from Baagoe.
    this.c = 1
    this.s0 = mash(' ')
    this.s1 = mash(' ')
    this.s2 = mash(' ')
    this.s0 -= mash(seed)
    if (this.s0 < 0) {
      this.s0 += 1
    }
    this.s1 -= mash(seed)
    if (this.s1 < 0) {
      this.s1 += 1
    }
    this.s2 -= mash(seed)
    if (this.s2 < 0) {
      this.s2 += 1
    }

    function mash(data: any) {
      data = String(data)
      for (let i = 0; i < data.length; i++) {
        n += data.charCodeAt(i)
        let h = 0.02519603282416938 * n
        n = h >>> 0
        h -= n
        h *= n
        n = h >>> 0
        h -= n
        n += h * 0x100000000 // 2^32
      }
      return (n >>> 0) * 2.3283064365386963e-10 // 2^-32
    }
  }

  next() {
    let { c, s0, s1, s2 } = this
    let t = 2091639 * s0 + c * 2.3283064365386963e-10 // 2^-32
    this.s0 = s1
    this.s1 = s2
    return (this.s2 = t - (this.c = t | 0))
  }

  copy(f: any, t: any) {
    t.c = f.c
    t.s0 = f.s0
    t.s1 = f.s1
    t.s2 = f.s2
    return t
  }
}

export type int = number

/**
 * Creates a new random number generator, optionally using the given seed.
 *
 * @param seed (Optional) the seed to be used.
 * @returns A new RNG method which returns floats from range `[0,1)`.
 */
export const createRng = (seed?: number) =>
  alea(seed?.toString()) as () => number

/**
 * Take a uniform [0,1) random number generator (RNG) function and turns it into an
 * integer RNG function.
 *
 * @param rand An RNG in the range of [0,1).
 * @returns An RNG that takes an integer range (from, until) as arguments and returns
 *          a random integer in the range [from, until).
 */
export const randInt = (rand: () => number) => (from: int, until: int) => {
  if (from >= until) {
    throw new Error('randInt(rng)(from,until): from must be less than until.')
  }
  const result = Math.floor(from + (until - from) * rand())
  if (result === until) {
    return result - 1
  }
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
  const normalizeDiv = nSamples - 1 || 1
  for (let i = nSamples - 1; i >= 0; i--) {
    const row = new Float64Array(nFeatures)
    row.fill(i / normalizeDiv)
    result.push(row)
  }

  // shuffle columns independently
  for (let i = nSamples - 1; i >= 1; i--) {
    let n = i + 1
    const ri = result[i]
    for (let j = nFeatures - 1; j >= 0; j--) {
      const k = _randInt(0, n)
      const rij = ri[j]
      ri[j] = result[k][j]
      result[k][j] = rij
    }
  }

  return result
}

interface Indexable<T> {
  [i: number]: T
  readonly length: number
}

export const shuffle =
  (rand: () => number) =>
  <T>(array: Indexable<T>) => {
    const _randInt = randInt(rand)
    for (let i = array.length; i > 1; i--) {
      const j = _randInt(0, i)
      const tmp = array[i - 1]
      array[i - 1] = array[j]
      array[j] = tmp
    }
  }
