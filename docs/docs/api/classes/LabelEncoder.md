---
id: "LabelEncoder"
title: "Class: LabelEncoder"
sidebar_label: "LabelEncoder"
sidebar_position: 0
custom_edit_url: null
---

Encode target labels with value between 0 and n_classes-1.

## Hierarchy

- `TransformerMixin`

  ↳ **`LabelEncoder`**

## Constructors

### constructor

• **new LabelEncoder**()

#### Overrides

TransformerMixin.constructor

#### Defined in

[preprocessing/encoders/label.encoder.ts:27](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L27)

## Properties

### $labels

• `Private` **$labels**: `Map`<`string` \| `number` \| `boolean`, `number`\>

#### Defined in

[preprocessing/encoders/label.encoder.ts:25](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L25)

## Accessors

### classes

• `get` **classes**(): `Map`<`string` \| `number` \| `boolean`, `number`\>

Get the mapping of classes to integers.

**`example`**
```
const encoder = new LabelEncoder()
encoder.fit(["a", "b", "c", "d"])
console.log(encoder.classes)
// {a: 0, b: 1, c: 2, d: 3}
```

#### Returns

`Map`<`string` \| `number` \| `boolean`, `number`\>

mapping of classes to integers.

#### Defined in

[preprocessing/encoders/label.encoder.ts:128](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L128)

___

### nClasses

• `get` **nClasses**(): `number`

Get the number of classes.

**`example`**
```
const encoder = new LabelEncoder()
encoder.fit(["a", "b", "c", "d"])
console.log(encoder.nClasses)
// 4
```

#### Returns

`number`

number of classes.

#### Defined in

[preprocessing/encoders/label.encoder.ts:113](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L113)

## Methods

### convertTo1DArray

▸ **convertTo1DArray**(`X`): `Iterable`<`string` \| `number` \| `boolean`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit1D` |

#### Returns

`Iterable`<`string` \| `number` \| `boolean`\>

#### Defined in

[preprocessing/encoders/label.encoder.ts:32](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L32)

___

### fit

▸ **fit**(`X`): [`LabelEncoder`](LabelEncoder)

Maps values to unique integer labels between 0 and n_classes-1.

**`example`**
```
const encoder = new LabelEncoder()
encoder.fit(["a", "b", "c", "d"])
```

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit1D` |

#### Returns

[`LabelEncoder`](LabelEncoder)

#### Defined in

[preprocessing/encoders/label.encoder.ts:51](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L51)

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

▸ **inverseTransform**(`X`): `any`[]

Inverse transform values back to original values.

**`example`**
```
const encoder = new LabelEncoder()
encoder.fit(["a", "b", "c", "d"])
console.log(encoder.inverseTransform([0, 1, 2, 3]))
// ["a", "b", "c", "d"]
```

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit1D` |

#### Returns

`any`[]

#### Defined in

[preprocessing/encoders/label.encoder.ts:92](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L92)

___

### transform

▸ **transform**(`X`): `Tensor1D`

Encode labels with value between 0 and n_classes-1.

**`example`**
```
const encoder = new LabelEncoder()
encoder.fit(["a", "b", "c", "d"])
console.log(encoder.transform(["a", "b", "c", "d"]))
// [0, 1, 2, 3]
```

#### Parameters

| Name | Type |
| :------ | :------ |
| `X` | `Scikit1D` |

#### Returns

`Tensor1D`

#### Defined in

[preprocessing/encoders/label.encoder.ts:72](https://github.com/dcrescim/scikit.js/blob/ecc4160/src/preprocessing/encoders/label.encoder.ts#L72)
