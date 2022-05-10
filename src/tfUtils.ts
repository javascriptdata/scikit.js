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

import { assert } from './typesUtils'

/**
 * Takes a `tf` instance and adds the `Unique` kernel to
 * it in case it uses a `tensorflow` backend and the `Unique`
 * kernel is missing. This polyfill becomes unecessary as soon
 * as the `Unique` kernel is added to `tfjs-node`.
 *
 * @see {@link https://github.com/tensorflow/tfjs/pull/5956}
 * @see {@link https://github.com/tensorflow/tfjs/issues/4595}
 *
 * @param tf The TFJS instance to be polyfilled.
 */
export function polyfillUnique(tf: any) {
  // TODO: remove this method as soon as tfjs-node supports tf.unique
  if (
    tf.engine().backendNames().includes('tensorflow') &&
    !tf.getKernel('Unique', 'tensorflow')
  ) {
    console.info('[scikit.js] Installing tfjs-node polyfill for tf.unique().')

    tf.registerKernel({
      kernelName: 'Unique',
      backendName: 'tensorflow',
      kernelFunc: (args: any) => {
        const x = args.inputs.x
        const backend = args.backend as any
        const { axis } = args.attrs as { axis: number }

        const axs = tf.tensor1d([axis], 'int32')

        const types = {
          float32: backend.binding.TF_FLOAT,
          float64: backend.binding.TF_DOUBLE,
          int32: backend.binding.TF_INT32,
          int64: backend.binding.TF_INT64,
          complex64: backend.binding.TF_COMPLEX64,
          bool: backend.binding.TF_BOOL,
          string: backend.binding.TF_STRING
        } as { [key: string]: number }

        assert(Object.keys(types).includes(x.dtype), 'Unexpected dtype.')

        try {
          const opAttrs = [
            {
              value: types[x.dtype],
              name: 'T',
              type: backend.binding.TF_ATTR_TYPE
            },
            {
              value: types.int32,
              name: 'Taxis',
              type: backend.binding.TF_ATTR_TYPE
            },
            {
              value: types.int32,
              name: 'out_idx',
              type: backend.binding.TF_ATTR_TYPE
            }
          ]
          return backend.executeMultipleOutputs(
            'UniqueV2',
            opAttrs,
            [x, axs],
            2
          )
        } finally {
          axs.dispose()
        }
      }
    })
  }
}
