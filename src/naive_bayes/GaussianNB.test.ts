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
import { GaussianNB } from './GaussianNB'
import { toObject, fromObject } from '../simpleSerializer'

describe('GaussianNB', function () {
  it('without priors', async () => {
    const X = [
      [0.1, 0.9],
      [0.3, 0.7],
      [0.9, 0.1],
      [0.8, 0.2],
      [0.81, 0.19]
    ]
    const y = [0, 0, 1, 1, 1]

    const model = new GaussianNB()

    await model.fit(X, y)
    const labels = model.predict(X)

    expect(labels.arraySync()).toEqual([0, 0, 1, 1, 1])
    expect(model.classes.size).toEqual(2)
    expect(model.means.length).toEqual(2)
    expect(model.variances.length).toEqual(2)
  })
  it('with priors', async () => {
    const X = [
      [0.1, 0.9],
      [0.3, 0.7],
      [0.9, 0.1],
      [0.8, 0.2],
      [0.81, 0.19]
    ]
    const y = [0, 0, 1, 1, 1]

    const model = new GaussianNB({ priors: [0.5, 0.5] })

    await model.fit(X, y)
    const labels = model.predict(X)

    expect(labels.arraySync()).toEqual([0, 0, 1, 1, 1])
  })
  it('with skewed priors', async () => {
    const X = [
      [0.1, 0.9],
      [0.3, 0.7],
      [0.9, 0.1],
      [0.8, 0.2],
      [0.81, 0.19]
    ]
    const y = [0, 0, 1, 1, 1]

    const model = new GaussianNB({ priors: [0.9, 0.1] })

    await model.fit(X, y)
    const labels = model.predict(X)

    expect(labels.arraySync()).toEqual([0, 0, 1, 1, 1])
  })
  it('with varSmoothing', async () => {
    const X = [
      [0.1, 0.9],
      [0.3, 0.7],
      [0.9, 0.1],
      [0.8, 0.2],
      [0.81, 0.19]
    ]
    const y = [0, 0, 1, 1, 1]

    const model = new GaussianNB({ priors: [0.5, 0.5], varSmoothing: 1.0 })

    await model.fit(X, y)
    const labels = model.predict(X)

    expect(labels.arraySync()).toEqual([0, 0, 1, 1, 1])
  })
  it('Should save and load Model', async () => {
    const X = [
      [0.1, 0.9],
      [0.3, 0.7],
      [0.9, 0.1],
      [0.8, 0.2],
      [0.81, 0.19]
    ]
    const y = [0, 0, 1, 1, 1]

    const model = new GaussianNB({ priors: [0.5, 0.5], varSmoothing: 1.0 })

    await model.fit(X, y)
    const labels = model.predict(X)

    const serializeModel = await toObject(model)
    const newModel = await fromObject(serializeModel)
    expect(newModel.predict(X).arraySync()).toEqual([0, 0, 1, 1, 1])
  })
})
