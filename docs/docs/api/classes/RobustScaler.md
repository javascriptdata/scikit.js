---
id: "RobustScaler"
title: "Class: RobustScaler"
sidebar_label: "RobustScaler"
sidebar_position: 0
custom_edit_url: null
---

## Hierarchy

- `TransformerMixin`

  ↳ **`RobustScaler`**

## Constructors

### constructor

• **new RobustScaler**()

#### Overrides

TransformerMixin.constructor

#### Defined in

[preprocessing/scalers/robust.scaler.ts:49](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/robust.scaler.ts#L49)

## Properties

### $center

• **$center**: `Tensor1D`

#### Defined in

[preprocessing/scalers/robust.scaler.ts:47](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/robust.scaler.ts#L47)

___

### $scale

• **$scale**: `Tensor1D`

#### Defined in

[preprocessing/scalers/robust.scaler.ts:46](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/robust.scaler.ts#L46)

## Methods

### fit

▸ **fit**(`X`): [`RobustScaler`](RobustScaler)

Fits a RobustScaler to the data

**`example`**
const scaler = new RobustScaler()
scaler.fit([1, 2, 3, 4, 5])
// RobustScaler {
//   $max: [5],
//   $min: [1]
// }

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

[`RobustScaler`](RobustScaler)

RobustScaler

#### Defined in

[preprocessing/scalers/robust.scaler.ts:68](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/robust.scaler.ts#L68)

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

[mixins.ts:3](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/mixins.ts#L3)

___

### inverseTransform

▸ **inverseTransform**(`X`): `Tensor2D`

Inverse transform the data using the fitted scaler

**`example`**
const scaler = new RobustScaler()
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

[preprocessing/scalers/robust.scaler.ts:117](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/robust.scaler.ts#L117)

___

### transform

▸ **transform**(`X`): `Tensor2D`

Transform the data using the fitted scaler

**`example`**
const scaler = new RobustScaler()
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

[preprocessing/scalers/robust.scaler.ts:100](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/robust.scaler.ts#L100)
