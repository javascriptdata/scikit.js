---
description: >-
  Contributing guide to scikit.js including set up guide, and a brief intro to folder structure
---

# Contributing Guide

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. 

For contributors familiar with open-source,  below is a quick guide to setting up scikit.js locally.  

```text
git clone https://github.com/opensource9ja/scikit.js.git
cd scikit.js
git checkout -b <your-branch-name>

yarn test:clean 
```

The following folders are available:
* `estimators`: All Machine learning algorithms.
* `model_selection`: Functions related to model selection.
* `preprocessing`: All functions for preprocessing data before, during and after training

The following files are available in the root directory:
* `index`: Entry file which exports all features. 
* `utils`: A collection of reusable utility functions. 
* `types`: A file for declaring Typescript types.


Some important scripts in the package.json file are:
* `test`: Run all test that satisfy the given pattern. Defaults to `test/**/**/*.test.ts` (All tests will be run)
* `test:clean` : Build the source to `dist` folder before running all test that satisfy the given pattern. This is useful when testing a new feature. 
* `build` : Compiles the src to the `dist` folder.
* `build:clean` : Cleans/Remove old folders before compiling the src to the `dist` folder.


## Code Style
### File names
File names must be all lowercase and compound names must be seperated by dots (.). E.g `label.encoder.ts`.

### Source file structure

Files consist of the following, in order:

 - License or copyright information, if present
 - ES import statements
 - The file’s source code 

 Example:

 ```javascript
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
 * @param number num1 
 * @param number num2 
 * @returns number 
 */
 getSum(num1, num2) {
   return num1 + num2

 ```

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

Sample documentation:

```javascript
 /**
 * Add two series of the same length
 * @param series1 The first Series. Defaults to []
 * @param series2 The second Series. Defaults to []
 * @returns Series
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

```

## **Writing tests**

We strongly encourage contributors to write tests for their code. Like many packages, danfojs uses mocha

All tests should go into the tests subdirectory and place in the corresponding module. The tests folder contains some current examples of tests, and we suggest looking to these for inspiration.

Below is the general Framework to write a test for each module.

```javascript
import { assert } from "chai"
import { addSeries } from '../../dist' //compiled build

describe("Name of the class|module", function(){
 
  it("name of the methods| expected result",function(){
    
       //write your test code here
       //use assert.{proprty} to test your code
   })

});
```

For a class with lots of methods.

```python
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

**1\)** Open the package.json 

**2\)** change the name of the test file to the file name you want. and don't forget the file is in the test folder

```python
"scripts": {
    "test": "....... tests/[sub_directory_name]/filename.test.ts",
```

**3\)**  run the test in clean mode

```python
yarn test:clean
```

Learn more about mocha [here](https://mochajs.org/)
