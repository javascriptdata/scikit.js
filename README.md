# scikit.js

JavaScript package for predictive data analysis and machine learning.

Aims to be a Typescript port of the [scikit-learn](https://scikit-learn.org) python library.

This library is for users who wish to train or deploy their models to JS environments (browser, mobile) but with a familiar API.

Generic math operations are powered by [Tensorflow.js](https://www.tensorflow.org/js) core layer for faster calculation.

Documentation site: [www.scikitjs.org](https://www.scikitjs.org)

<img width="396" alt="135392530-81ed4901-10fc-4d74-9fec-da8c968573f5" src="https://user-images.githubusercontent.com/29900845/137105982-f1a51ad5-9adb-46c3-9dfc-d3a23e36d93f.png" />

# Installation

### Frontend Users

For use with modern bundlers in a frontend application, simply

```js
yarn add scikitjs
```

### Backend Users

For Node.js users who wish to bind to the Tensorflow C++ library, simply

```js
yarn add scikitjs-node
```

### Script src

For those that wish to use script src tags, simply

```js
<script src="https://unpkg.com/scikitjs/dist/scikit.min.js"></script>
```

## Simple Example

```js
import { LinearRegression } from 'scikitjs'

const lr = LinearRegression({ fitIntercept: false })
const X = [[1], [2]] // 2D Matrix with a single column vector
const y = [10, 20]

await lr.fit(X, y)

lr.predict([[3, 4]]) // roughly [30, 40]
console.log(lr.coef_)
console.log(lr.intercept_)
```

## Contributing Guide

See guide [here](https://github.com/opensource9ja/scikit.js/blob/dev/CONTRIBUTING_GUIDE.md)
