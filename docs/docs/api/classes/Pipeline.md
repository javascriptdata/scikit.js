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

[pipeline/pipeline.ts:13](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L13)

## Properties

### steps

• **steps**: `Bunch`[]

#### Defined in

[pipeline/pipeline.ts:12](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L12)

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

[pipeline/pipeline.ts:17](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L17)

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

[pipeline/pipeline.ts:61](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L61)

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

[pipeline/pipeline.ts:40](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L40)

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

[pipeline/pipeline.ts:50](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L50)

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

[pipeline/pipeline.ts:30](https://github.com/dcrescim/scikit.js/blob/ae98366/scikitjs-node/src/pipeline/pipeline.ts#L30)
