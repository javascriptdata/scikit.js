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

/**
 * Container for results of a nearest or farthest neighbor search.
 *
 * @type A The address type.
 * @type V The type of value stored with each address.
 */
export interface Neighbor<A, V> extends NeighborhoodEntry<A, V> {
  distance: number
}

/**
 * Used as parameters for the construction of {@link Neighborhood}
 * instances. Also returned as items during iteration over the content
 * of {@link Neighborhood} instances.
 *
 * @type A The address type.
 * @type V The type of value stored with each address.
 */
export interface NeighborhoodEntry<A, V> {
  address: A
  value: V
}

/**
 * Default constructor parameters for a {@link Neighborhood}.
 *
 * @type A The address type.
 * @type V The type of value stored with each address.
 */
export interface NeighborhoodParams<A, V> {
  metric: (x: A, y: A) => number
  entries?: Iterable<NeighborhoodEntry<A, V>>
}

/**
 * A collections of address-value-pairs that allows (reasonably) fast
 * search of nearest an farthest neighbors search for a given address.
 * The distance between addresses is computed by some metric. Different
 * implementations may support different types of metrics.
 *
 * @type A The address type.
 * @type V The type of value stored with each address.
 */
export interface Neighborhood<A, V> extends Iterable<NeighborhoodEntry<A, V>> {
  nearest(address: A): IterableIterator<Neighbor<A, V>>
  farthest(address: A): IterableIterator<Neighbor<A, V>>
}
