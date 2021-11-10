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

preprocessing/encoders/ordinal.encoder.ts:32

## Properties

### $labels

• **$labels**: `Map`<`string` \| `number` \| `boolean`, `number`\>[]

#### Defined in

preprocessing/encoders/ordinal.encoder.ts:30

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

preprocessing/encoders/ordinal.encoder.ts:63

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

### loopOver2DArrayToSetLabels

▸ **loopOver2DArrayToSetLabels**(`array2D`): `void`

#### Parameters

| Name | Type |
| :------ | :------ |
| `array2D` | `any` |

#### Returns

`void`

#### Defined in

preprocessing/encoders/ordinal.encoder.ts:37

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

preprocessing/encoders/ordinal.encoder.ts:69

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

preprocessing/encoders/ordinal.encoder.ts:94
