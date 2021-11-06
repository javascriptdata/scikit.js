---
id: "OrdinalEncoder"
title: "Class: OrdinalEncoder"
sidebar_label: "OrdinalEncoder"
sidebar_position: 0
custom_edit_url: null
---

Fits a OrdinalEncoder to the data.

**`example`**
```js
const encoder = new OrdinalEncoder()
encoder.fit(["a", "b", "c"])
```

## Hierarchy

- `TransformerMixin`

  ↳ **`OrdinalEncoder`**

## Constructors

### constructor

• **new OrdinalEncoder**()

#### Overrides

TransformerMixin.constructor

#### Defined in

[preprocessing/encoders/ordinal.encoder.ts:31](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/ordinal.encoder.ts#L31)

## Properties

### $labels

• **$labels**: `Map`<`string` \| `number` \| `boolean`, `number`\>[]

#### Defined in

[preprocessing/encoders/ordinal.encoder.ts:29](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/ordinal.encoder.ts#L29)

## Methods

### fit

▸ **fit**(`X`, `y?`): [`OrdinalEncoder`](OrdinalEncoder)

Fits a OrdinalEncoder to the data.

**`example`**
```js
const encoder = new OrdinalEncoder()
encoder.fit(["a", "b", "c"])
```

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |
| `y?` | `Scikit1D` |

#### Returns

[`OrdinalEncoder`](OrdinalEncoder)

OrdinalEncoder

#### Defined in

[preprocessing/encoders/ordinal.encoder.ts:62](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/ordinal.encoder.ts#L62)

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

### loopOver2DArrayToSetLabels

▸ **loopOver2DArrayToSetLabels**(`array2D`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `array2D` | `any` |

#### Returns

`void`

#### Defined in

[preprocessing/encoders/ordinal.encoder.ts:36](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/ordinal.encoder.ts#L36)

___

### loopOver2DArrayToUseLabels

▸ **loopOver2DArrayToUseLabels**(`array2D`): `number`[][]

#### Parameters

| Name | Type |
| :------ | :------ |
| `array2D` | `any` |

#### Returns

`number`[][]

#### Defined in

[preprocessing/encoders/ordinal.encoder.ts:68](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/ordinal.encoder.ts#L68)

___

### transform

▸ **transform**(`X`, `y?`): `Tensor2D`

Encodes the data using the fitted OneHotEncoder.

**`example`**
```js
const encoder = new OneHotEncoder()
encoder.fit(["a", "b", "c"])
encoder.transform(["a", "b", "c"])
```

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |
| `y?` | `Scikit1D` |

#### Returns

`Tensor2D`

#### Defined in

[preprocessing/encoders/ordinal.encoder.ts:93](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/ordinal.encoder.ts#L93)
