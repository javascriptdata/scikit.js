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
import { BaseNaiveBayes } from "./baseNaiveBayes";

/**
 * Gaussian Naive Bayes classifier
 *
 * @example
 * ```js
 * import { GaussianNB } from 'scikitjs'
 *
 * const clf = new GaussianNB({ priors: [0.5, 0.5] })
   const X = [
     [0.1, 0.9],
     [0.3, 0.7],
     [0.9, 0.1],
     [0.8, 0.2],
     [0.81, 0.19]
   ]
   const y = [0, 0, 1, 1, 1]

   const model = new GaussianNB({})
   await model.fit(X, y)

   clf.predict([
     [0.1, 0.9],
     [0.01, 0.99]
   ]) // [0, 1]


 * ```
 *
 */
export class GaussianNB extends BaseNaiveBayes {
  protected kernel(features: tf.Tensor2D, mean: tf.Tensor1D, variance: tf.Tensor1D): tf.Tensor1D {
    return tf.tidy(() => {
        return tf.sub(features, mean.expandDims(0))
            .pow(2)
            .div(variance.expandDims(0).mul(-2))
            .exp()
            .div(variance.mul(2 * Math.PI).expandDims(0).sqrt())
            .prod(1) as tf.Tensor1D
    })
  }
}