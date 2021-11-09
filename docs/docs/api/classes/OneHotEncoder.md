---
id: "OneHotEncoder"
title: "Class: OneHotEncoder"
sidebar_label: "OneHotEncoder"
sidebar_position: 0
custom_edit_url: null
---

Fits a OneHotEncoder to the data.

**`example`**
```js
const encoder = new OneHotEncoder()
encoder.fit(["a", "b", "c"])
```

## Hierarchy

- `TransformerMixin`

  ↳ **`OneHotEncoder`**

## Constructors

### constructor

• **new OneHotEncoder**()

#### Overrides

TransformerMixin.constructor

#### Defined in

[preprocessing/encoders/one.hot.encoder.ts:38](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L38)

## Properties

### $labels

• **$labels**: `Map`<`string` \| `number` \| `boolean`, `number`\>[]

#### Defined in

[preprocessing/encoders/one.hot.encoder.ts:36](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L36)

## Methods

### fit

▸ **fit**(`X`, `y?`): [`OneHotEncoder`](OneHotEncoder)

Fits a OneHotEncoder to the data.

**`example`**
```js
const encoder = new OneHotEncoder()
encoder.fit(["a", "b", "c"])
```

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |
| `y?` | `Scikit1D` |

#### Returns

[`OneHotEncoder`](OneHotEncoder)

OneHotEncoder

#### Defined in

[preprocessing/encoders/one.hot.encoder.ts:69](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L69)

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

[mixins.ts:3](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/mixins.ts#L3)

___

### inverseTransform

▸ **inverseTransform**(`X`): `any`[]

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Tensor2D` |

#### Returns

`any`[]

#### Defined in

[preprocessing/encoders/one.hot.encoder.ts:110](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L110)

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

[preprocessing/encoders/one.hot.encoder.ts:43](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L43)

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

[preprocessing/encoders/one.hot.encoder.ts:75](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L75)

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

[preprocessing/encoders/one.hot.encoder.ts:100](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/preprocessing/encoders/one.hot.encoder.ts#L100)
