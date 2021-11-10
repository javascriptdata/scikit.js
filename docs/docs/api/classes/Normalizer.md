---
id: "Normalizer"
title: "Class: Normalizer"
sidebar_label: "Normalizer"
sidebar_position: 0
custom_edit_url: null
---

## Hierarchy

- `TransformerMixin`

  ↳ **`Normalizer`**

## Constructors

### constructor

• **new Normalizer**(`__namedParameters?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `__namedParameters` | `NormalizerParams` |

#### Overrides

TransformerMixin.constructor

#### Defined in

preprocessing/scalers/normalizer.ts:34

## Properties

### norm

• **norm**: `string`

#### Defined in

preprocessing/scalers/normalizer.ts:33

## Methods

### fit

▸ **fit**(`X`): [`Normalizer`](Normalizer)

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

[`Normalizer`](Normalizer)

MinMaxScaler

#### Defined in

preprocessing/scalers/normalizer.ts:52

___

### fitTransform

▸ **fitTransform**(`X`): `any`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `any` |

#### Returns

`any`

#### Inherited from

TransformerMixin.fitTransform

#### Defined in

mixins.ts:3

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

#### Defined in

preprocessing/scalers/normalizer.ts:67
