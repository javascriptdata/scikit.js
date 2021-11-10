import { onesLike, Tensor, tidy, where } from '@tensorflow/tfjs-core'
// import * as tf from '@tensorflow/tfjs-node'
import { Iterable } from './types'
import { assert } from './types.utils'

import { tf } from './globals'
/*
In creating the preprocessors, I wanted functions that computed the min, max, mean,
etc... but that also ignored NaN values. The min / max functions that come from
Tensorflow don't support ignoring NaN's, so we perform some magic to get the result.

After running some tests, I've concluded that the fastest way to
get the min, mean, median, most_frequent, etc... is to not copy the input array.
So the "fastest" way to perform any action is to simply loop over the initial
array if the user passes in an array, or to use Tensor methods if the user
passes in a Tensor.

The "copying" of the array into a tf Tensor is way slower than simply writing the
for loop.

So below we have simple mean, max, min, functions that work with arrays, or tensors
that ignore NaNs
*/

//////////////////////////////////////////////////////////////////////////
// Min Functions
//////////////////////////////////////////////////////////////////////////
export function simpleMin<T extends Iterable<number | string | boolean>>(
  arr: T,
  ignoreNaN?: boolean
): number | string | boolean {
  assert(
    arr.length > 0,
    `Array ${arr} must have length greater than 0 in order to find a minimum element`
  )
  let min = arr[0]
  for (let i = 0; i < arr.length; i++) {
    let current = arr[i]
    if (ignoreNaN && current === NaN) {
      continue
    }
    if (current < min) {
      min = current
    }
  }
  return min
}

export function tensorMin(
  tensor: Tensor,
  axis: number,
  ignoreNaN: boolean
): Tensor {
  if (ignoreNaN) {
    return tidy(() => where(tensor.isNaN(), Infinity, tensor).min(axis))
  }
  return tensor.min(axis)
}

//////////////////////////////////////////////////////////////////////////
// Max Functions
//////////////////////////////////////////////////////////////////////////

export function simpleMax<T extends Iterable<number | string | boolean>>(
  arr: T,
  ignoreNaN?: boolean
): number | string | boolean {
  assert(
    arr.length > 0,
    `Array ${arr} must have length greater than 0 in order to find a maximum element`
  )
  let max = arr[0]
  for (let i = 0; i < arr.length; i++) {
    let current = arr[i]
    if (ignoreNaN && current === NaN) {
      continue
    }
    if (current > max) {
      max = current
    }
  }
  return max
}

export function tensorMax(
  tensor: Tensor,
  axis: number,
  ignoreNaN?: boolean
): Tensor {
  if (ignoreNaN) {
    return tidy(() => where(tensor.isNaN(), -Infinity, tensor).max(axis))
  }
  return tensor.min(axis)
}

//////////////////////////////////////////////////////////////////////////
// Sum Functions
//////////////////////////////////////////////////////////////////////////

export function simpleSum<T extends Iterable<number | boolean>>(
  arr: T,
  ignoreNaN?: boolean
): number {
  let total = 0
  for (let i = 0; i < arr.length; i++) {
    const current = arr[i]
    if (ignoreNaN && current === NaN) {
      continue
    }
    total += Number(current)
  }
  return total
}

export function tensorSum(tensor: Tensor, axis: number, ignoreNaN?: boolean) {
  if (ignoreNaN) {
    return tidy(() => where(tensor.isNaN(), 0, tensor).sum(axis))
  }
  return tensor.sum(axis)
}

//////////////////////////////////////////////////////////////////////////
// Count Functions
//////////////////////////////////////////////////////////////////////////

export function simpleCount<T extends Iterable<number | string | boolean>>(
  arr: T,
  ignoreNaN?: boolean
): number {
  if (!ignoreNaN) {
    return arr.length
  }

  let count = 0
  for (let i = 0; i < arr.length; i++) {
    const current = arr[i]
    if (current === NaN) {
      continue
    }
    count += 1
  }
  return count
}

export function tensorCount(
  tensor: Tensor,
  axis: number,
  ignoreNaN?: boolean
) {
  if (ignoreNaN) {
    return tidy(() => tf.logicalNot(tensor.isNaN()).sum(axis))
  }

  // Could definitely do this faster
  return onesLike(tensor).sum(axis)
}

//////////////////////////////////////////////////////////////////////////
// Mean Functions
//////////////////////////////////////////////////////////////////////////

export function simpleMean<T extends Iterable<number | boolean>>(
  arr: T
): number {
  let count = simpleCount(arr)
  if (count === 0) {
    return 0
  }
  let sum = simpleSum(arr)
  return sum / count
}

export function tensorMean(
  tensor: Tensor,
  axis: number,
  ignoreNaN?: boolean,
  safe?: boolean
) {
  if (!ignoreNaN) {
    return tensor.mean(axis)
  }

  if (safe) {
    return tidy(() =>
      tensorSum(tensor, axis, ignoreNaN).div(
        turnZerosToOnes(tensorCount(tensor, axis, ignoreNaN))
      )
    )
  }

  return tidy(() =>
    tensorSum(tensor, axis, ignoreNaN).div(
      tensorCount(tensor, axis, ignoreNaN)
    )
  )
}

//////////////////////////////////////////////////////////////////////////
// Std Functions
//////////////////////////////////////////////////////////////////////////

export function tensorStd(tensor: Tensor, dim: number, ignoreNaN?: boolean) {
  assert(
    Boolean(ignoreNaN),
    'We only need to call this function when ignoreNaN is true'
  )

  return tidy(() => {
    const mean = tensorMean(tensor, dim, ignoreNaN)
    const countNaN = tensorCount(tensor, dim, ignoreNaN)

    const numerator = tensorSum(tensor.sub(mean).square(), dim, ignoreNaN)

    // Choose biased variance over unbiased to match sklearn
    const denominator = turnZerosToOnes(countNaN)

    return numerator.div(denominator).sqrt()
  })
}

export function turnZerosToOnes(tensor: Tensor) {
  return tidy(() => {
    const zeros = tf.zerosLike(tensor)
    const booleanAddition = tensor.equal(zeros)
    return tensor.add(booleanAddition)
  })
}
