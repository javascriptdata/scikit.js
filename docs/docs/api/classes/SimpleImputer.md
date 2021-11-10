---
id: "SimpleImputer"
title: "Class: SimpleImputer"
sidebar_label: "SimpleImputer"
sidebar_position: 0
custom_edit_url: null
---

## Hierarchy

- `TransformerMixin`

  ↳ **`SimpleImputer`**

## Constructors

### constructor

• **new SimpleImputer**(`__namedParameters?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `__namedParameters` | `SimpleImputerParams` |

#### Overrides

TransformerMixin.constructor

#### Defined in

impute/simple.imputer.ts:59

## Properties

### fillValue

• **fillValue**: `Tensor1D`

#### Defined in

impute/simple.imputer.ts:56

___

### missingValues

• **missingValues**: `undefined` \| ``null`` \| `string` \| `number`

#### Defined in

impute/simple.imputer.ts:55

___

### strategy

• **strategy**: `Strategy`

#### Defined in

impute/simple.imputer.ts:57

## Methods

### fit

▸ **fit**(`X`): [`SimpleImputer`](SimpleImputer)

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

[`SimpleImputer`](SimpleImputer)

#### Defined in

impute/simple.imputer.ts:70

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

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`Tensor2D`

#### Defined in

impute/simple.imputer.ts:117
