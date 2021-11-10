---
id: "DummyRegressor"
title: "Class: DummyRegressor"
sidebar_label: "DummyRegressor"
sidebar_position: 0
custom_edit_url: null
---

## Hierarchy

- `PredictorMixin`

  ↳ **`DummyRegressor`**

## Constructors

### constructor

• **new DummyRegressor**(`__namedParameters?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `__namedParameters` | `DummyRegressorParams` |

#### Overrides

PredictorMixin.constructor

#### Defined in

dummy/dummy.regressor.ts:50

## Properties

### $fill

• **$fill**: `number`

#### Defined in

dummy/dummy.regressor.ts:47

___

### $strategy

• **$strategy**: `string`

#### Defined in

dummy/dummy.regressor.ts:48

## Methods

### fit

▸ **fit**(`X`, `y`): [`DummyRegressor`](DummyRegressor)

Fit a DummyClassifier to the data.

**`example`**
const dummy = new DummyClassifier()
dummy.fit([[1,1], [2,2], [3,3]],[1, 2, 3])

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `X` | `ScikitVecOrMatrix` | Array, Tensor, DataFrame or Series object |
| `y` | `Scikit1D` | Array, Series object |

#### Returns

[`DummyRegressor`](DummyRegressor)

DummyClassifier

#### Defined in

dummy/dummy.regressor.ts:65

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

PredictorMixin.fitPredict

#### Defined in

mixins.ts:10

___

### predict

▸ **predict**(`X`): `any`[]

Predicts response on given example data

**`example`**
const dummy = new DummyRegressor('median')
dummy.fit([1, 3, 3, 10, 20])
dummy.predict([1, 2, 3, 4, 5])
// [3, 3, 3, 3, 3]

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `X` | `ScikitVecOrMatrix` | Array, Tensor, DataFrame or Series object |

#### Returns

`any`[]

Array, Tensor, DataFrame or Series object

#### Defined in

dummy/dummy.regressor.ts:96
