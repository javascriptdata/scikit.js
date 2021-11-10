---
id: "Pipeline"
title: "Class: Pipeline"
sidebar_label: "Pipeline"
sidebar_position: 0
custom_edit_url: null
---

## Constructors

### constructor

• **new Pipeline**(`__namedParameters?`)

#### Parameters

| Name | Type |
| :------ | :------ |
| `__namedParameters` | `PipelineParams` |

#### Defined in

pipeline/pipeline.ts:13

## Properties

### steps

• **steps**: `Bunch`[]

#### Defined in

pipeline/pipeline.ts:12

## Methods

### fit

▸ **fit**(`X`, `y`): `Promise`<[`Pipeline`](Pipeline)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |
| `y` | `Scikit1D` |

#### Returns

`Promise`<[`Pipeline`](Pipeline)\>

#### Defined in

pipeline/pipeline.ts:17

___

### fitPredict

▸ **fitPredict**(`X`, `y`): `any`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |
| `y` | `Scikit1D` |

#### Returns

`any`

#### Defined in

pipeline/pipeline.ts:61

___

### fitTransform

▸ **fitTransform**(`X`, `y`): `Scikit2D`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |
| `y` | `Scikit1D` |

#### Returns

`Scikit2D`

#### Defined in

pipeline/pipeline.ts:40

___

### predict

▸ **predict**(`X`): `any`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`any`

#### Defined in

pipeline/pipeline.ts:50

___

### transform

▸ **transform**(`X`): `Scikit2D`

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit2D` |

#### Returns

`Scikit2D`

#### Defined in

pipeline/pipeline.ts:30
