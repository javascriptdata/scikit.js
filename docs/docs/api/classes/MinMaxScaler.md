---
id: "MinMaxScaler"
title: "Class: MinMaxScaler"
sidebar_label: "MinMaxScaler"
sidebar_position: 0
custom_edit_url: null
---

Transform features by scaling each feature to a given range.
This estimator scales and translates each feature individually such
that it is in the given range on the training set, e.g. between the maximum and minimum value.

## Hierarchy

- `TransformerMixin`

  ↳ **`MinMaxScaler`**

## Implements

- `Transformer`

## Constructors

### constructor

• **new MinMaxScaler**()

#### Overrides

TransformerMixin.constructor

#### Defined in

preprocessing/scalers/min.max.scaler.ts:36

## Properties

### $min

• **$min**: `Tensor1D`

#### Defined in

preprocessing/scalers/min.max.scaler.ts:34

___

### $scale

• **$scale**: `Tensor1D`

#### Defined in

preprocessing/scalers/min.max.scaler.ts:33

## Methods

### fit

▸ **fit**(`X`): [`MinMaxScaler`](MinMaxScaler)

Fits a MinMaxScaler to the data

**`example`**
const scaler = new MinMaxScaler()
scaler.fit([1, 2, 3, 4, 5])
// MinMaxScaler {
//   $max: [5],
//   $min: [1]
// }

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

[`MinMaxScaler`](MinMaxScaler)

MinMaxScaler

#### Implementation of

Transformer.fit

#### Defined in

preprocessing/scalers/min.max.scaler.ts:55

___

### fitTransform

▸ **fitTransform**(`X`): `any`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `any` |

#### Returns

`any`

#### Implementation of

Transformer.fitTransform

#### Inherited from

TransformerMixin.fitTransform

#### Defined in

mixins.ts:3

___

### inverseTransform

▸ **inverseTransform**(`X`): `Tensor2D`

Inverse transform the data using the fitted scaler

**`example`**
const scaler = new MinMaxScaler()
scaler.fit([1, 2, 3, 4, 5])
scaler.inverseTransform([0, 0.25, 0.5, 0.75, 1])
// [1, 2, 3, 4, 5]

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`Tensor2D`

Array, Tensor, DataFrame or Series object

#### Defined in

preprocessing/scalers/min.max.scaler.ts:99

___

### transform

▸ **transform**(`X`): `Tensor2D`

Transform the data using the fitted scaler

**`example`**
const scaler = new MinMaxScaler()
scaler.fit([1, 2, 3, 4, 5])
scaler.transform([1, 2, 3, 4, 5])
// [0, 0.25, 0.5, 0.75, 1]

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`Tensor2D`

Array, Tensor, DataFrame or Series object

#### Implementation of

Transformer.transform

#### Defined in

preprocessing/scalers/min.max.scaler.ts:82
