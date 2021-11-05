---
id: "DummyClassifier"
title: "Class: DummyClassifier"
sidebar_label: "DummyClassifier"
sidebar_position: 0
custom_edit_url: null
---

Creates an estimator that guesses a class label based on simple rules.

## Hierarchy

- `PredictorMixin`

  ↳ **`DummyClassifier`**

## Constructors

### constructor

• **new DummyClassifier**(`__namedParameters?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `__namedParameters` | `DummyClassifierParams` |

#### Overrides

PredictorMixin.constructor

#### Defined in

[dummy/dummy.classifier.ts:43](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/dummy/dummy.classifier.ts#L43)

## Properties

### $fill

• **$fill**: `number`

#### Defined in

[dummy/dummy.classifier.ts:39](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/dummy/dummy.classifier.ts#L39)

___

### $strategy

• **$strategy**: `string`

#### Defined in

[dummy/dummy.classifier.ts:40](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/dummy/dummy.classifier.ts#L40)

___

### $uniques

• **$uniques**: `number`[] \| `string`[]

#### Defined in

[dummy/dummy.classifier.ts:41](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/dummy/dummy.classifier.ts#L41)

## Methods

### fit

▸ **fit**(`X`, `y`): [`DummyClassifier`](DummyClassifier)

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

[`DummyClassifier`](DummyClassifier)

DummyClassifier

#### Defined in

[dummy/dummy.classifier.ts:62](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/dummy/dummy.classifier.ts#L62)

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

[mixins.ts:10](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/mixins.ts#L10)

___

### predict

▸ **predict**(`X`): `any`[]

Predicts response on given example data

**`example`**
const dummy = new DummyClassifier()
dummy.fit([1, 3, 3, 4, 5])
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

[dummy/dummy.classifier.ts:93](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/dummy/dummy.classifier.ts#L93)
