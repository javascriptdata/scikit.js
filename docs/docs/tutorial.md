---
sidebar_position: 1
---

# Tutorial

Let's discover **Scikit.js in less than 5 minutes**.

## Getting Started

Get started by **installing the library as well as it's dependencies**.

```shell
npm install scikitjs @tensorflow/tfjs
```

or

```shell
yarn add scikitjs @tensorflow/tfjs
```

## Build a model

Build a simple Linear Regression

```js
// import tensorflow and register it as the backend
import * as tf from '@tensorflow/tfjs'
import { LinearRegression, setBackend } from 'scikitjs'
setBackend(tf)

// Perform a linear regression
let X = [
  [2, 3],
  [1, 4],
  [5, 7]
]
let y = [10, 14, 20]

let lr = new LinearRegression()
await lr.fit(X, y)
console.log(lr.coef)
```
