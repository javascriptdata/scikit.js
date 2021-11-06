---
id: "StandardScaler"
title: "Class: StandardScaler"
sidebar_label: "StandardScaler"
sidebar_position: 0
custom_edit_url: null
---

Standardize features by removing the mean and scaling to unit variance.
The standard score of a sample x is calculated as: `z = (x - u) / s`,
where `u` is the mean of the training samples, and `s` is the standard deviation of the training samples.

## Hierarchy

- `TransformerMixin`

  ↳ **`StandardScaler`**

## Constructors

### constructor

• **new StandardScaler**()

#### Overrides

TransformerMixin.constructor

#### Defined in

[preprocessing/scalers/standard.scaler.ts:32](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/standard.scaler.ts#L32)

## Properties

### $mean

• **$mean**: `Tensor`<`Rank`\>

#### Defined in

[preprocessing/scalers/standard.scaler.ts:30](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/standard.scaler.ts#L30)

___

### $std

• **$std**: `Tensor`<`Rank`\>

#### Defined in

[preprocessing/scalers/standard.scaler.ts:29](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/standard.scaler.ts#L29)

## Methods

### fit

▸ **fit**(`X`): [`StandardScaler`](StandardScaler)

Fit a StandardScaler to the data.

**`example`**
const scaler = new StandardScaler()
scaler.fit([1, 2, 3, 4, 5])

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

[`StandardScaler`](StandardScaler)

StandardScaler

#### Defined in

[preprocessing/scalers/standard.scaler.ts:46](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/standard.scaler.ts#L46)

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
const scaler = new StandardScaler()
scaler.fit([1, 2, 3, 4, 5])
scaler.transform([1, 2, 3, 4, 5])
// [0.0, 0.0, 0.0, 0.0, 0.0]
scaler.inverseTransform([0.0, 0.0, 0.0, 0.0, 0.0])
// [1, 2, 3, 4, 5]

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`Tensor2D`

Array, Tensor, DataFrame or Series object

#### Defined in

[preprocessing/scalers/standard.scaler.ts:86](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/standard.scaler.ts#L86)

___

### transform

▸ **transform**(`X`): `Tensor2D`

Transform the data using the fitted scaler

**`example`**
const scaler = new StandardScaler()
scaler.fit([1, 2, 3, 4, 5])
scaler.transform([1, 2, 3, 4, 5])
// [0.0, 0.0, 0.0, 0.0, 0.0]

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`Tensor2D`

Array, Tensor, DataFrame or Series object

#### Defined in

[preprocessing/scalers/standard.scaler.ts:67](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/scalers/standard.scaler.ts#L67)
