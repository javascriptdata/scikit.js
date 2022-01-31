---
description: >-
  Contributing guide to scikit.js including set up guide, and a brief intro to folder structure
---

# Contributing Guide

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

For contributors familiar with open-source, below is a quick guide to setting up scikit.js locally.

```text
git clone https://github.com/javascriptdata/scikit.js.git
cd scikit.js
git checkout -b <your-branch-name>

npm run test:clean
```

The following repo has 4 main directories:

- `scikitjs-browser`: Contains build/test commands for the browser version of scikitjs
- `scikitjs-node`: Contains build/test commands for the node version of scikitjs
- `docs`: Contains a docusaurus site which builds the `scikitjs.org` site. Has blogs/tutorials/apis documentation
- `shared`: Contains all the code that is shared between browser and node versions of this library.

For anyone creating Estimators or writing scikit-learn functions, you'll likely be spending your time in the `shared/lib` directory. It contains a directory-like structure that matches the scikit-learn directory structure. So there is a `cluster` directory, and a `model_selection` directory, etc.

The following files are available in the `shared/lib` directory:

- `index`: Entry file which exports all features.
- `utils`: A collection of reusable utility functions.
- `types`: A file for declaring Typescript types.

Some important scripts in the package.json file are:

- `test:clean` : Build both browser and node versions, and runs all tests against them
- `build` : Builds both the browser and node versions of this library
- `build:docs`: Builds a local version of the site `scikitjs.org`

## Code Style

### File names

File names must be all lowerCamelCase names that specify the function or class that is inside. E.g `labelEncoder.ts` houses the class `LabelEncoder`. `trainTestSplit.ts` contains the function `trainTestSplit`.

### Source file structure

Files consist of the following, in order:

- License or copyright information, if present
- ES import statements
- The file’s source code

Example:

````typescript
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

import table from 'table-data'
import me from 'you'
import { cat, dog, eagle } from '../animals'

/**
 * Returns the sum of two numbers
 * * @example
 * ```js
 * const result = getSum(2, 3)
 * console.log(result) // 5
 * ```
 */
function getSum(num1: number, num2: number): number {
  return num1 + num2
}
````

### Naming Convention

#### Class names

Class, interface, record, and typedef names are written in UpperCamelCase e.g `ImageProcessor`.
Type names are typically nouns or noun phrases. For example, Request, ImmutableList, or VisibilityMode.

#### Method names

Method names are written in lowerCamelCase e.g `addNum`. Names for private methods must start with a dollar sign e.g `$startAddition`, and should be declared as private.

Method names are typically verbs or verb phrases. For example, `sendMessage` or `$stopProcess`. Getter and setter methods for properties are never required, but if they are used they should be named `getFoo` (or optionally `isFoo` or `hasFoo` for booleans), or `setFoo(value)` for setters.

#### Constant names

Constant names use `CONSTANT_CASE`: all uppercase letters, with words separated by underscores.

### JSDOC Guidelines

Documentation helps clarify what a function or a method is doing. It also gives insight to users of the function or methods on what parameters to pass in and know what the function will return.

Whenever you are writing a class or a function, it's best to start with a high-level description and then go directly into an example usage. The @param are not needed because Typedoc picks up the parameters and returns types directly from the type signature.

Sample documentation:

````typescript
/**
 * Add two series of the same length
 * @example
 * ```
 * import { addSeries } from "scikit.js"
 *
 * const newSeries = addSeries(Sf1, Sf2)
 * newSeries.shape.print()
 * // [10, 4]
 * ```
 */
const addSeries = (series1, series2) => {
  //DO something here
  return new Series()
}
````

JSDOCs attached to the class or functions are automatically converted into the API documentation hosted at [https://scikitjs.org/docs/api](https://scikitjs.org/docs/api).

## **Writing tests**

We strongly encourage contributors to write tests for their code. Like many packages, [danfojs](https://danfo.jsdata.org/) uses [mocha](https://mochajs.org/).

All tests should go into the file suffixed by `.test.ts` and place in the corresponding module. The test files contain some current examples of tests (e.g. `kmeans.test.ts`), and we suggest looking to these for inspiration.

Below is the general framework to write a test for each module.

```typescript
import { assert } from 'chai'
import { addSeries } from './addSeries' //compiled build

describe('Name of the class|module', function () {
  it('name of the methods| expected result', function () {
    //write your test code here
    //use assert.{property} to test your code
  })
})
```

For a class with lots of methods.

```typescript
import { assert } from "chai"
import { DataFrame } from '../../src/core/frame'

describe("Name of the class|module", function(){

 describe("method name 1", function(){

   it("expected result",function(){

        //write your test code here
        //use assert.{proprty} to test your code
    })
  })

  describe("method name 2", function(){

   it("expected result",function(){

        //write your test code here
        //use assert.{proprty} to test your code
    })
  })
  .......
});
```

### **Running the test case**

To run the test for the module/file you created/edited,

**1\)** Change `describe` to `describe.only` at the top of the test file

**2\)** If you are in the shared directory, you can run the test case against the node library by typing

```bash
npm run test:node
```

You can also test your code against the browser version by running

```bash
npm run test:browser
```

Note that running tests against the browser is usually a bit slower because we build the browser bundle. I usually test against node for quick iterations, and then at the end test against the browser.

Learn more about mocha [here](https://mochajs.org/)
