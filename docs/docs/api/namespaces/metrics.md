---
id: "metrics"
title: "Namespace: metrics"
sidebar_label: "metrics"
sidebar_position: 0
custom_edit_url: null
---

## Functions

### accuracyScore

▸ **accuracyScore**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:42

___

### confusionMatrix

▸ **confusionMatrix**(`labels`, `predictions`): `number`[][]

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`[][]

#### Defined in

metrics/metrics.ts:162

___

### hingeLoss

▸ **hingeLoss**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:122

___

### huberLoss

▸ **huberLoss**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:131

___

### logLoss

▸ **logLoss**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:140

___

### meanAbsoluteError

▸ **meanAbsoluteError**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:89

___

### meanSquaredError

▸ **meanSquaredError**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:101

___

### meanSquaredLogError

▸ **meanSquaredLogError**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:110

___

### precisionScore

▸ **precisionScore**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:54

___

### r2Score

▸ **r2Score**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:72

___

### recallScore

▸ **recallScore**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:63

___

### rocAucScore

▸ **rocAucScore**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:175

___

### zeroOneLoss

▸ **zeroOneLoss**(`labels`, `predictions`): `number`

#### Parameters

| Name | Type |
| :------ | :------ |
| `labels` | `Scikit1D` |
| `predictions` | `Scikit1D` |

#### Returns

`number`

#### Defined in

metrics/metrics.ts:149
