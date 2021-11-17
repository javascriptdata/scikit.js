---
sidebar_position: 1
---

# Tutorial

Let's discover **Scikit.js in less than 5 minutes**.

## Getting Started

Get started by **installing the library**.

```shell
npm install scikitjs
```

or

```shell
yarn add scikitjs
```

## Build a model

Build a simple Linear Regression

```js
import { LinearRegression } from 'scikitjs'

let X = [
  [2, 3],
  [1, 4],
  [5, 7]
]
let y = [10, 14, 20]

let lr = new LinearRegression()
await lr.fit(X, y)
console.log(lr.coef_)
```
