---
sidebar_position: 4
---

# Contributing

Want to add your favorite estimator in JS? Want to make Javascript a better language for machine learning? Awesome!
In this article, we'll add an example Estimator to this project. In doing so, we'll learn

1. How this project is setup
2. How to create / document your new estimator
3. What are the types in this library
4. What are the deploy targets

## DummyRegressor

Let's make a [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). It's an Estimator that predicts a `y` value based on simple rules.
So for example, if you pass a `strategy` of "mean", then it will look at the response variable (`y`), compute the mean, and return that value on any input. Here's some example usage.

```js
import { DummyRegressor } from 'scikitjs'

let myReg = new DummyRegressor({ strategy: 'mean' })
await myReg.fit(
  [
    [1, 2],
    [3, 4],
    [5, 6]
  ],
  [10, 20, 30]
) // The mean is 20, so anything that we call predict on should return 20

const expect = myReg.predict([
  [2, 10],
  [1, 10]
]) // returns a 1D Tensor which is basically [20, 20]
```

Based on the scikit-learn [DummyRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html), we see that the `DummyRegressor` supports `strategy`, `constant`, and `quantile` as constructor inputs. For simplicity let's only support `strategy` and `constant` in this walkthrough.

Moreover, the only `strategy` values that we will support will be `"mean"` and `"constant"`.

## First Pass

Without further ado, let's create a class

```typescript
class DummyRegressor {
  constructor(args) {
    // does stuff
  }

  fit(X, y) {
    // fits stuff
  }

  predict(X) {
    // predicts stuff
  }
}
```

The first order of business, is how do we pass constructor arguments to our class? We could pass those arguments positionally, like `constructor(strategy, constant)`, or we can pass it as an object like `constructor({strategy, constant})`. As explained in [Coming from python](/python.md), I chose to do objects because there are some Estimators that take in a large number of options with sane defaults like [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier) and I didn't want users to have to type in all the defaults.

## What types though?

The second order of business, is what exactly are the types that we pass into `fit` and `predict`? Looking at the [Estimators](https://scikit-learn.org/stable/developers/develop.html) in scikit-learn it's clear that the `X` is a 2D Array, and the `y` is a 1D Array. We also want to support the `DataFrame`, and `Series` objects from `danfo` (the pandas equivalent in JS), and `Tensor2D` and `Tensor1D` (Tensors are the JS version of numpy arrays. They ship with Tensorflow.).

After some deliberation, here are the types that we use in this library to represent the following things.

```typescript
// The Types that Scikit uses
export type TypedArray = Float32Array | Int32Array | Uint8Array
export type ScikitLike1D = TypedArray | number[] | boolean[] | string[]
export type ScikitLike2D = TypedArray[] | number[][] | boolean[][] | string[][]
export type Scikit1D = ScikitLike1D | Tensor1D | Series
export type Scikit2D = ScikitLike2D | Tensor2D | DataFrame
export type ScikitVecOrMatrix = Scikit1D | Scikit2D
```

## What we have so far

After putting in the proper types for the arguments, we have the following

```typescript
class DummyRegressor {
  constructor({ strategy = 'mean', constant }) {
    // does stuff
  }

  fit(X: Scikit2D, y: Scikit1D) {
    // fits stuff
  }

  predict(X: Scikit2D) {
    // predicts stuff
  }
}
```

Note also, that `fit` should return a reference to the class itself, and `predict` usually returns a numpy array, which in javascript land is a `Tensor2D`. Let's save our constructor args, and add those typings.

```typescript
class DummyRegressor {
  constructor({ strategy = 'mean', constant }) {
    this.strategy = strategy
    this.constant = constant
  }

  fit(X: Scikit2D, y: Scikit1D): DummyRegressor {
    // fit stuff
  }

  predict(X: Scikit2D): Tensor1D {
    // predicts stuff
  }
}
```

Now let's write the `fit` function. If the user has set the strategy to "mean", than we will construct a Tensor1D from the `y` array, and calculate the mean of it. If the strategy is "constant", then we do nothing, because the `constant` class property already contains the fill value that we will use for prediction.

```typescript
class DummyRegressor {
  constructor({ strategy = 'mean', constant }) {
    this.strategy = strategy
    this.constant = constant
  }

  fit(X: Scikit2D, y: Scikit1D): DummyRegressor {
    const newY = convertToNumericTensor1D(y)

    if (this.strategy === 'mean') {
      this.constant = newY.mean().dataSync()[0]
      return this
    }

    // constant case
    return this
  }

  predict(X: Scikit2D): Tensor1D {
    // predicts stuff
  }
}
```

As you can see, there are a bunch of nice utility functions that we can use to convert user input into formats more amenable to number crunching. In this case, I use `convertToNumericTensor1D` to turn `y` into a Tensor1D.

Then I call mean on that `Tensor` and then use the [Tensorflow API](https://js.tensorflow.org/api/latest/) `dataSync` function to get the actual returned value as a single number instead of a `Tensor1D`.

Let's finish writing `predict`. In this case, I'm simply going to take the input, convert it to a 2D Tensor, and then get the number of rows. From there, I'm going to simply create a Tensor1D and fill it with whatever is in my `constant` attribute. That looks like this.

```typescript
class DummyRegressor {
  constructor({ strategy = 'mean', constant }) {
    this.strategy = strategy
    this.constant = constant
  }

  fit(X: Scikit2D, y: Scikit1D): DummyRegressor {
    const newY = convertToNumericTensor1D(y)

    if (this.strategy === 'mean') {
      this.constant = newY.mean().dataSync()[0]
      return this
    }

    // constant case
    return this
  }

  predict(X: Scikit2D): Tensor1D {
    let newData = convertToNumericTensor2D(X)
    let length = newData.shape[0]
    return tensor1d(Array(length).fill(this.constant))
  }
}
```

And there you go! Certainly there is more to do. We haven't talked about `assert`ing on bad user input, or supporting all the options for DummyRegressor (both constructor args, and class methods), but it's a start. And we can start testing.

I think the next question is "Where do I put this code in the repo?". And in order to answer that question we need to talk about the project setup and deploy targets.

## Project setup

This single project needs to serve both frontend and backend needs. It has 3 deploy targets.

1. It needs to be useful in a modern frontend framework which includes a bundler. This means that you need to be able to

```js
yarn add scikitjs
```

and have it just work.

2. It needs to be compatible with script tags. Someone should be able to just

```js
<script src="https://unpkg.com/scikitjs/dist/web/index.min.js"></script>
```

3. It needs to be compatible with backend Node.js environments, and use the C++ Tensorflow.js bindings for speed improvements there.

```js
yarn add scikitjs
```

And then use it easily with

```js
import { LinearRegression } from 'scikitjs/node'
```

Because there are multiple deploy environments (frontend / backend and eventually gpu), we have multiple output directories. We build an esm (ES Modules), cjs (Commonjs modules), (ES5 build) for older browsers, and a full-fledged script tag bundle which bundles up tensorflow, and other dependencies.

The repo basically looks like this.

```
scikitjs
│   README.md
│   package.json
│
└───src
│   │   package.json
│   └───shared
│       │    globals.ts
|
|   └───shared-esm
|       |    globals.ts
|
|   └───shared-node
|       |    globals.ts
│
└───docs
│   │   package.json
│   └───src
│       │    globals.ts
│
```

The `src` directory contains all of the important algorithmic code. The build scripts swap the `shared/globals.ts` file for other `shared-*/globals.ts` files that have different versions of our dependencies to build different versions of the library (esm, cjs, etc).

## There ya have it

So that's the basic idea. There are things I skipped over (testing, using mixins, asserting on bad input), but this will get the gravy train rolling.
