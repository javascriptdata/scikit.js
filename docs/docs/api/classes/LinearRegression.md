---
id: "LinearRegression"
title: "Class: LinearRegression"
sidebar_label: "LinearRegression"
sidebar_position: 0
custom_edit_url: null
---

## Hierarchy

- `SGD`

  ↳ **`LinearRegression`**

## Constructors

### constructor

• **new LinearRegression**(`__namedParameters?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `__namedParameters` | `LinearRegressionParams` |

#### Overrides

SGD.constructor

#### Defined in

estimators/linear.model.ts:44

## Properties

### denseLayerArgs

• **denseLayerArgs**: `DenseLayerArgs`

#### Inherited from

SGD.denseLayerArgs

#### Defined in

estimators/sgd.linear.ts:102

___

### isClassification

• **isClassification**: `boolean`

#### Inherited from

SGD.isClassification

#### Defined in

estimators/sgd.linear.ts:105

___

### model

• **model**: `Sequential`

#### Inherited from

SGD.model

#### Defined in

estimators/sgd.linear.ts:99

___

### modelCompileArgs

• **modelCompileArgs**: `ModelCompileArgs`

#### Inherited from

SGD.modelCompileArgs

#### Defined in

estimators/sgd.linear.ts:101

___

### modelFitArgs

• **modelFitArgs**: `ModelFitArgs`

#### Inherited from

SGD.modelFitArgs

#### Defined in

estimators/sgd.linear.ts:100

___

### oneHot

• **oneHot**: [`OneHotEncoder`](OneHotEncoder)

#### Inherited from

SGD.oneHot

#### Defined in

estimators/sgd.linear.ts:106

## Accessors

### coef\_

• `get` **coef_**(): `Tensor1D` \| `Tensor2D`

Similar to scikit-learn, this returns the coefficients of our linear model.
The return type is a 1D matrix (technically a Tensor1D) if we predict a single output.
It's a 2D matrix (Tensor2D) if we predict a regression task with multiple outputs or
a classification task with multiple class labels.

**`example`**

lr = new LinearRegression()
await lr.fit(X, [1,2,3]);
lr.coef_
// => tensor1d([[ 1.2, 3.3, 1.1, 0.2 ]])

await lr.fit(X, [ [1,2], [3,4], [5,6] ]);
lr.coef_
// => tensor2d([ [1.2, 3.3], [3.4, 5.6], [4.5, 6.7] ])

#### Returns

`Tensor1D` \| `Tensor2D`

Returns the coefficients.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.coef\_

#### Defined in

estimators/sgd.linear.ts:366

___

### intercept\_

• `get` **intercept_**(): `number` \| `Tensor1D`

Similar to scikit-learn, this returns the intercept of our linear model.
The return type is always a Tensor1D (a vector).
Normally we'd just return a single number but in the case
of multiple regression (multiple output targets) we'd need
a vector to store all the intercepts,

**`example`**

lr = new LinearRegression()
await lr.fit(X, [1,2,3]);
lr.intercept_
// => 4.5

lr = new LinearRegression()
await lr.fit(X, [ [1,2,3], [4,5,6] ]);
lr.intercept_
// => tensor1d([1.2, 2.3])

#### Returns

`number` \| `Tensor1D`

Returns the intercept.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.intercept\_

#### Defined in

estimators/sgd.linear.ts:402

## Methods

### fit

▸ **fit**(`X`, `y`): `Promise`<`SGD`\>

Similar to scikit-learn, this trains a model to predict y, from X.
Even in the case where we predict a single output vector,
the predictions are a 2D matrix (albeit a single column in a 2D Matrix).

This is to facilitate the case where we predict multiple targets, or in the case
of classification where we are predicting a 2D Matrix of probability class labels.

**`example`**

lr = new LinearRegression()
await lr.fit(X, y);
// lr model weights have been updated

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `X` | `Scikit2D` | The 2DTensor / 2D Array that you wish to use as a training matrix |
| `y` | `ScikitVecOrMatrix` | Either 1D or 2D array / Tensor that you wish to predict |

#### Returns

`Promise`<`SGD`\>

Returns the predictions.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.fit

#### Defined in

estimators/sgd.linear.ts:198

___

### fitPredict

▸ **fitPredict**(`X`, `y`): `any`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `any` |
| `y` | `any` |

#### Returns

`any`

#### Inherited from

SGD.fitPredict

#### Defined in

mixins.ts:10

___

### getParams

▸ **getParams**(): `SGDParams`

Similar to scikit-learn, this returns the object of configuration params for SGD

**`example`**

lr = new LinearRegression()
lr.getParams()
// =>
{
modelCompileArgs: {
optimizer: train.adam(0.1),
loss: losses.meanSquaredError,
metrics: ['mse'],
},
modelFitArgs: {
batchSize: 32,
epochs: 1000,
verbose: 0,
callbacks: [callbacks.earlyStopping({ monitor: 'mse', patience: 50 })],
},
denseLayerArgs: {
units: 1,
useBias: true,
}
}

#### Returns

`SGDParams`

Returns an object of configuration params.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.getParams

#### Defined in

estimators/sgd.linear.ts:274

___

### importModel

▸ **importModel**(`params`): `SGD`

This aims to be a bridge to scikit-learn Estimators, where users can train
models over in scikit-learn and then ship the coefficients into the proper
Estimator on the Scikit.js side. This can be useful if the python version is faster
to train, but we still need a JS version because we wish to ship to mobile or browsers.

**`example`**

lr = new LinearRegression()
lr.importModel({coef_ : [1.2, 2.3], intercept_: 10.0});
// lr model weights have been updated

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `params` | `Object` | The object that contains the model parameters, coef_, and intercept_ that we need for our model. |
| `params.coef_` | `number`[] | - |
| `params.intercept_` | `number` | - |

#### Returns

`SGD`

Returns the predictions.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.importModel

#### Defined in

estimators/sgd.linear.ts:235

___

### initializeModel

▸ **initializeModel**(`X`, `y`, `weightsTensors?`): `void`

Creates the tensorflow model. Because the model contains only
one dense layer, we must pass the inputShape to that layer.
That inputShape is only known at "runtime" ie... when we call `fit(X, y)`
that first time. The inputShape is effectively `X.shape[1]`

This function runs after that first call to fit or when pass in modelWeights.
That can come up if we train a model in python, and simply want to copy over the
weights to this JS version so we can deploy on browsers / phones.

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `X` | `Tensor2D` | `undefined` |
| `y` | `Tensor1D` \| `Tensor2D` | `undefined` |
| `weightsTensors` | `Tensor`<`Rank`\>[] | `[]` |

#### Returns

`void`

#### Inherited from

SGD.initializeModel

#### Defined in

estimators/sgd.linear.ts:159

___

### initializeModelForClassification

▸ **initializeModelForClassification**(`y`): `Tensor2D`

#### Parameters

| Name | Type |
| :------ | :------ |
| `y` | `Tensor1D` \| `Tensor2D` |

#### Returns

`Tensor2D`

#### Inherited from

SGD.initializeModelForClassification

#### Defined in

estimators/sgd.linear.ts:126

___

### predict

▸ **predict**(`X`): `Tensor1D` \| `Tensor2D`

Similar to scikit-learn, this returns a Tensor2D (2D Matrix) of predictions.
Even in the case where we predict a single output vector,
the predictions are a 2D matrix (albeit a single column in a 2D Matrix).

This is to facilitate the case where we predict multiple targets, or in the case
of classification where we are predicting a 2D Matrix of probability class labels.

**`example`**

lr = new LinearRegression()
await lr.fit(X, y);
lr.predict(X)
// => tensor2d([[ 4.5, 10.3, 19.1, 0.22 ]])

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `X` | `Scikit2D` | The 2DTensor / 2D Array that you wish to run through your model and make predictions. |

#### Returns

`Tensor1D` \| `Tensor2D`

Returns the predictions.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.predict

#### Defined in

estimators/sgd.linear.ts:331

___

### setParams

▸ **setParams**(`params`): `SGD`

Similar to scikit-learn, this returns the object of configuration params for SGD

**`example`**

lr = new LinearRegression()
lr.setParams({
modelFitArgs: {
batchSize: 100,
epochs: -1,
verbose: 1,
})

#### Parameters

| Name | Type |
| :------ | :------ |
| `params` | `SGDParams` |

#### Returns

`SGD`

Returns an object of configuration params.

We use a LinearRegression in the example below because it provides
defaults for the SGD

#### Inherited from

SGD.setParams

#### Defined in

estimators/sgd.linear.ts:301
